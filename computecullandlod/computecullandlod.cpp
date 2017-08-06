/*
* Vulkan Example - Compute shader culling and LOD using indirect rendering
*
* Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h> 
#include <vector>
#include <random>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <vulkan/vulkan.hpp>
#include "vulkanexamplebase.h"
#include "VulkanBuffer.hpp"
#include "VulkanModel.hpp"
#include "frustum.hpp"

#define VERTEX_BUFFER_BIND_ID 0
#define INSTANCE_BUFFER_BIND_ID 1
#define ENABLE_VALIDATION false

// Total number of objects (^3) in the scene
#if defined(__ANDROID__)
#define OBJECT_COUNT 32
#else
#define OBJECT_COUNT 64
#endif

#define MAX_LOD_LEVEL 5

class VulkanExample : public VulkanExampleBase
{
public:
	bool fixedFrustum = false;

	struct {
		vk::PipelineVertexInputStateCreateInfo inputState;
		std::vector<vk::VertexInputBindingDescription> bindingDescriptions;
		std::vector<vk::VertexInputAttributeDescription> attributeDescriptions;
	} vertices;

	// Vertex layout for the models
	vks::VertexLayout vertexLayout = vks::VertexLayout({
		vks::VERTEX_COMPONENT_POSITION,
		vks::VERTEX_COMPONENT_NORMAL,
		vks::VERTEX_COMPONENT_COLOR,
	});

	struct {
		vks::Model lodObject;
	} models;

	// Per-instance data block
	struct InstanceData {
		glm::vec3 pos;
		float scale;
	};

	// Contains the instanced data
	vks::Buffer instanceBuffer;
	// Contains the indirect drawing commands
	vks::Buffer indirectCommandsBuffer;
	vks::Buffer indirectDrawCountBuffer;

	// Indirect draw statistics (updated via compute)
	struct {
		uint32_t drawCount;						// Total number of indirect draw counts to be issued
		uint32_t lodCount[MAX_LOD_LEVEL + 1];	// Statistics for number of draws per LOD level (written by compute shader)
	} indirectStats;

	// Store the indirect draw commands containing index offsets and instance count per object
	std::vector<vk::DrawIndexedIndirectCommand> indirectCommands;

	struct {
		glm::mat4 projection;
		glm::mat4 modelview;
		glm::vec4 cameraPos;
		glm::vec4 frustumPlanes[6];
	} uboScene;

	struct {
		vks::Buffer scene;
	} uniformData;

	struct {
		vk::Pipeline plants;
	} pipelines;

	vk::PipelineLayout pipelineLayout;
	vk::DescriptorSet descriptorSet;
	vk::DescriptorSetLayout descriptorSetLayout;

	// Resources for the compute part of the example
	struct {
		vks::Buffer lodLevelsBuffers;				// Contains index start and counts for the different lod levels
		vk::Queue queue;								// Separate queue for compute commands (queue family may differ from the one used for graphics)
		vk::CommandPool commandPool;					// Use a separate command pool (queue family may differ from the one used for graphics)
		vk::CommandBuffer commandBuffer;				// Command buffer storing the dispatch commands and barriers
		vk::Fence fence;								// Synchronization fence to avoid rewriting compute CB if still in use
		vk::Semaphore semaphore;						// Used as a wait semaphore for graphics submission
		vk::DescriptorSetLayout descriptorSetLayout;	// Compute shader binding layout
		vk::DescriptorSet descriptorSet;				// Compute shader bindings
		vk::PipelineLayout pipelineLayout;			// Layout of the compute pipeline
		vk::Pipeline pipeline;						// Compute pipeline for updating particle positions
	} compute;

	// View frustum for culling invisible objects
	vks::Frustum frustum;

	uint32_t objectCount = 0;

	VulkanExample() : VulkanExampleBase(ENABLE_VALIDATION)
	{
		enableTextOverlay = true;
		title = "Vulkan Example - Compute cull and lod";
		camera.type = Camera::CameraType::firstperson;
		camera.setPerspective(60.0f, (float)width / (float)height, 0.1f, 512.0f);
		camera.setTranslation(glm::vec3(0.5f, 0.0f, 0.0f));
		camera.movementSpeed = 5.0f;
		memset(&indirectStats, 0, sizeof(indirectStats));
	}

	~VulkanExample()
	{
		device.destroyPipeline(pipelines.plants);
		device.destroyPipelineLayout(pipelineLayout);
		device.destroyDescriptorSetLayout(descriptorSetLayout);
		models.lodObject.destroy();
		instanceBuffer.destroy();
		indirectCommandsBuffer.destroy();
		uniformData.scene.destroy();
		indirectDrawCountBuffer.destroy();
		compute.lodLevelsBuffers.destroy();
		device.destroyPipelineLayout(compute.pipelineLayout);
		device.destroyDescriptorSetLayout(compute.descriptorSetLayout);
		device.destroyPipeline(compute.pipeline);
		device.destroyFence(compute.fence);
		device.destroyCommandPool(compute.commandPool);
		device.destroySemaphore(compute.semaphore);
	}

	virtual void getEnabledFeatures()
	{
		// Enable multi draw indirect if supported
		if (deviceFeatures.multiDrawIndirect) {
			enabledFeatures.multiDrawIndirect = VK_TRUE;
		}
	}

	void buildCommandBuffers()
	{
		vk::CommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();

		vk::ClearValue clearValues[2];
		clearValues[0].color = { { 0.18f, 0.27f, 0.5f, 0.0f } };
		clearValues[1].depthStencil = { 1.0f, 0 };

		vk::RenderPassBeginInfo renderPassBeginInfo = vks::initializers::renderPassBeginInfo();
		renderPassBeginInfo.renderPass = renderPass;
		renderPassBeginInfo.renderArea.extent.width = width;
		renderPassBeginInfo.renderArea.extent.height = height;
		renderPassBeginInfo.clearValueCount = 2;
		renderPassBeginInfo.pClearValues = clearValues;

		for (uint32_t i = 0; i < drawCmdBuffers.size(); ++i)
		{
			// Set target frame buffer
			renderPassBeginInfo.framebuffer = frameBuffers[i];

			drawCmdBuffers[i].begin(cmdBufInfo);

			drawCmdBuffers[i].beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);

			vk::Viewport viewport = vks::initializers::viewport((float)width, (float)height, 0.0f, 1.0f);
			drawCmdBuffers[i].setViewport(0, viewport);

			vk::Rect2D scissor = vks::initializers::rect2D(width, height, 0, 0);
			vkCmdSetScissor(drawCmdBuffers[i], 0, 1, &scissor);

			vk::DeviceSize offsets[1] = { 0 };
			drawCmdBuffers[i].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSet, nullptr);

			// Mesh containing the LODs
			drawCmdBuffers[i].bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.plants);
			drawCmdBuffers[i].bindVertexBuffers(VERTEX_BUFFER_BIND_ID, models.lodObject.vertices.buffer, offsets);
			drawCmdBuffers[i].bindVertexBuffers(INSTANCE_BUFFER_BIND_ID, 1, instanceBuffer.buffer, offsets);
			
			drawCmdBuffers[i].bindIndexBuffer(models.lodObject.indices.buffer, 0, vk::IndexType::eUint32);

			if (vulkanDevice->features.multiDrawIndirect)
			{
				drawCmdBuffers[i].drawIndexedIndirect(indirectCommandsBuffer.buffer, 0, indirectStats.drawCount, sizeof(vk::DrawIndexedIndirectCommand));
			}
			else
			{
				// If multi draw is not available, we must issue separate draw commands
				for (uint32_t j = 0; j < indirectCommands.size(); j++)
				{
					drawCmdBuffers[i].drawIndexedIndirect(indirectCommandsBuffer.buffer, j * sizeof(vk::DrawIndexedIndirectCommand), 1, sizeof(vk::DrawIndexedIndirectCommand));
				}
			}	

			drawCmdBuffers[i].endRenderPass();

			drawCmdBuffers[i].end();
		}
	}

	void loadAssets()
	{
		models.lodObject.loadFromFile(getAssetPath() + "models/suzanne_lods.dae", vertexLayout, 0.1f, vulkanDevice, queue);
	}

	void setupVertexDescriptions()
	{
		vertices.bindingDescriptions.resize(2);

		// Binding 0: Per vertex
		vertices.bindingDescriptions[0] =
			vks::initializers::vertexInputBindingDescription(VERTEX_BUFFER_BIND_ID, vertexLayout.stride(), vk::VertexInputRate::eVertex);

		// Binding 1: Per instance
		vertices.bindingDescriptions[1] = 
			vks::initializers::vertexInputBindingDescription(INSTANCE_BUFFER_BIND_ID, sizeof(InstanceData), vk::VertexInputRate::eInstance);

		// Attribute descriptions
		// Describes memory layout and shader positions
		vertices.attributeDescriptions.clear();

		// Per-Vertex attributes
		// Location 0 : Position
		vertices.attributeDescriptions.push_back(
			vks::initializers::vertexInputAttributeDescription(
				VERTEX_BUFFER_BIND_ID,
				0,
				vk::Format::eR32G32B32Sfloat,
				0)
			);
		// Location 1 : Normal
		vertices.attributeDescriptions.push_back(
			vks::initializers::vertexInputAttributeDescription(
				VERTEX_BUFFER_BIND_ID,
				1,
				vk::Format::eR32G32B32Sfloat,
				sizeof(float) * 3)
			);
		// Location 2 : Color
		vertices.attributeDescriptions.push_back(
			vks::initializers::vertexInputAttributeDescription(
				VERTEX_BUFFER_BIND_ID,
				2,
				vk::Format::eR32G32B32Sfloat,
				sizeof(float) * 6)
			);

		// Instanced attributes
		// Location 4: Position
		vertices.attributeDescriptions.push_back(
			vks::initializers::vertexInputAttributeDescription(
				INSTANCE_BUFFER_BIND_ID, 4, vk::Format::eR32G32B32Sfloat, offsetof(InstanceData, pos))
			);
		// Location 5: Scale
		vertices.attributeDescriptions.push_back(
			vks::initializers::vertexInputAttributeDescription(
				INSTANCE_BUFFER_BIND_ID, 5, vk::Format::eR32Sfloat, offsetof(InstanceData, scale))
			);

		vertices.inputState = vks::initializers::pipelineVertexInputStateCreateInfo();
		vertices.inputState.vertexBindingDescriptionCount = static_cast<uint32_t>(vertices.bindingDescriptions.size());
		vertices.inputState.pVertexBindingDescriptions = vertices.bindingDescriptions.data();
		vertices.inputState.vertexAttributeDescriptionCount = static_cast<uint32_t>(vertices.attributeDescriptions.size());
		vertices.inputState.pVertexAttributeDescriptions = vertices.attributeDescriptions.data();
	}

	void buildComputeCommandBuffer()
	{
		vk::CommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();

		compute.commandBuffer.begin(cmdBufInfo);

		// Add memory barrier to ensure that the indirect commands have been consumed before the compute shader updates them
		vk::BufferMemoryBarrier bufferBarrier = vks::initializers::bufferMemoryBarrier();
		bufferBarrier.buffer = indirectCommandsBuffer.buffer;
		bufferBarrier.size = indirectCommandsBuffer.descriptor.range;
		bufferBarrier.srcAccessMask = vk::AccessFlagBits::eIndirectCommandRead;						
		bufferBarrier.dstAccessMask = vk::AccessFlagBits::eShaderWrite;																																																								
		bufferBarrier.srcQueueFamilyIndex = vulkanDevice->queueFamilyIndices.graphics;			
		bufferBarrier.dstQueueFamilyIndex = vulkanDevice->queueFamilyIndices.compute;			

		vkCmdPipelineBarrier(
			compute.commandBuffer,
			vk::PipelineStageFlagBits::eDrawIndirect,
			vk::PipelineStageFlagBits::eComputeShader,
			VK_FLAGS_NONE,
			0, nullptr,
			1, &bufferBarrier,
			0, nullptr);

		compute.commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, compute.pipeline);
		compute.commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, compute.pipelineLayout, 0, compute.descriptorSet, nullptr);

		// Dispatch the compute job
		// The compute shader will do the frustum culling and adjust the indirect draw calls depending on object visibility. 
		// It also determines the lod to use depending on distance to the viewer.
		vkCmdDispatch(compute.commandBuffer, objectCount / 16, 1, 1);

		// Add memory barrier to ensure that the compute shader has finished writing the indirect command buffer before it's consumed
		bufferBarrier.srcAccessMask = vk::AccessFlagBits::eShaderWrite;
		bufferBarrier.dstAccessMask = vk::AccessFlagBits::eIndirectCommandRead;
		bufferBarrier.buffer = indirectCommandsBuffer.buffer;
		bufferBarrier.size = indirectCommandsBuffer.descriptor.range;
		bufferBarrier.srcQueueFamilyIndex = vulkanDevice->queueFamilyIndices.compute;
		bufferBarrier.dstQueueFamilyIndex = vulkanDevice->queueFamilyIndices.graphics;

		vkCmdPipelineBarrier(
			compute.commandBuffer,
			vk::PipelineStageFlagBits::eComputeShader,
			vk::PipelineStageFlagBits::eDrawIndirect,
			VK_FLAGS_NONE,
			0, nullptr,
			1, &bufferBarrier,
			0, nullptr);

		// todo: barrier for indirect stats buffer?

		vkEndCommandBuffer(compute.commandBuffer);
	}

	void setupDescriptorPool()
	{
		// Example uses one ubo 
		std::vector<vk::DescriptorPoolSize> poolSizes =
		{
			vks::initializers::descriptorPoolSize(vk::DescriptorType::eUniformBuffer, 2),
			vks::initializers::descriptorPoolSize(vk::DescriptorType::eStorageBuffer, 4)
		};

		vk::DescriptorPoolCreateInfo descriptorPoolInfo =
			vks::initializers::descriptorPoolCreateInfo(
				static_cast<uint32_t>(poolSizes.size()),
				poolSizes.data(),
				2);

		descriptorPool = device.createDescriptorPool(descriptorPoolInfo);
	}

	void setupDescriptorSetLayout()
	{
		std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings =
		{
			// Binding 0: Vertex shader uniform buffer
			vks::initializers::descriptorSetLayoutBinding(
				vk::DescriptorType::eUniformBuffer,
				vk::ShaderStageFlagBits::eVertex,
				0),
		};

		vk::DescriptorSetLayoutCreateInfo descriptorLayout =
			vks::initializers::descriptorSetLayoutCreateInfo(
				setLayoutBindings.data(),
				static_cast<uint32_t>(setLayoutBindings.size()));

		descriptorSetLayout = device.createDescriptorSetLayout(descriptorLayout);

		vk::PipelineLayoutCreateInfo pPipelineLayoutCreateInfo =
			vks::initializers::pipelineLayoutCreateInfo(
				&descriptorSetLayout,
				1);

		pipelineLayout = device.createPipelineLayout(pPipelineLayoutCreateInfo);
	}

	void setupDescriptorSet()
	{
		vk::DescriptorSetAllocateInfo allocInfo =
			vks::initializers::descriptorSetAllocateInfo(
				descriptorPool,
				&descriptorSetLayout,
				1);

		descriptorSet = device.allocateDescriptorSets(allocInfo)[0];

		std::vector<vk::WriteDescriptorSet> writeDescriptorSets =
		{
			// Binding 0: Vertex shader uniform buffer
			vks::initializers::writeDescriptorSet(
			descriptorSet,
				vk::DescriptorType::eUniformBuffer,
				0,
				&uniformData.scene.descriptor),
		};

		device.updateDescriptorSets(writeDescriptorSets);
	}

	void preparePipelines()
	{
		vk::PipelineInputAssemblyStateCreateInfo inputAssemblyState =
			vks::initializers::pipelineInputAssemblyStateCreateInfo(
				vk::PrimitiveTopology::eTriangleList,
				0,
				VK_FALSE);

		vk::PipelineRasterizationStateCreateInfo rasterizationState =
			vks::initializers::pipelineRasterizationStateCreateInfo(
				vk::PolygonMode::eFill,
				vk::CullModeFlagBits::eBack,
				vk::FrontFace::eClockwise,
				0);

		vk::PipelineColorBlendAttachmentState blendAttachmentState =
			vks::initializers::pipelineColorBlendAttachmentState(
				0xf,
				VK_FALSE);

		vk::PipelineColorBlendStateCreateInfo colorBlendState =
			vks::initializers::pipelineColorBlendStateCreateInfo(
				1,
				&blendAttachmentState);

		vk::PipelineDepthStencilStateCreateInfo depthStencilState =
			vks::initializers::pipelineDepthStencilStateCreateInfo(
				VK_TRUE,
				VK_TRUE,
				vk::CompareOp::eLessOrEqual);

		vk::PipelineViewportStateCreateInfo viewportState =
			vks::initializers::pipelineViewportStateCreateInfo(1, 1, 0);

		vk::PipelineMultisampleStateCreateInfo multisampleState =
			vks::initializers::pipelineMultisampleStateCreateInfo(
				vk::SampleCountFlagBits::e1,
				0);

		std::vector<vk::DynamicState> dynamicStateEnables = {
			vk::DynamicState::eViewport,
			vk::DynamicState::eScissor
		};
		vk::PipelineDynamicStateCreateInfo dynamicState =
			vks::initializers::pipelineDynamicStateCreateInfo(
				dynamicStateEnables.data(),
				static_cast<uint32_t>(dynamicStateEnables.size()),
				0);

		vk::GraphicsPipelineCreateInfo pipelineCreateInfo =
			vks::initializers::pipelineCreateInfo(
				pipelineLayout,
				renderPass,
				0);

		std::array<vk::PipelineShaderStageCreateInfo, 2> shaderStages;

		pipelineCreateInfo.pVertexInputState = &vertices.inputState;
		pipelineCreateInfo.pInputAssemblyState = &inputAssemblyState;
		pipelineCreateInfo.pRasterizationState = &rasterizationState;
		pipelineCreateInfo.pColorBlendState = &colorBlendState;
		pipelineCreateInfo.pMultisampleState = &multisampleState;
		pipelineCreateInfo.pViewportState = &viewportState;
		pipelineCreateInfo.pDepthStencilState = &depthStencilState;
		pipelineCreateInfo.pDynamicState = &dynamicState;
		pipelineCreateInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
		pipelineCreateInfo.pStages = shaderStages.data();

		// Indirect (and instanced) pipeline for the plants
		shaderStages[0] = loadShader(getAssetPath() + "shaders/computecullandlod/indirectdraw.vert.spv", vk::ShaderStageFlagBits::eVertex);
		shaderStages[1] = loadShader(getAssetPath() + "shaders/computecullandlod/indirectdraw.frag.spv", vk::ShaderStageFlagBits::eFragment);
		pipelines.plants = device.createGraphicsPipelines(pipelineCache, pipelineCreateInfo)[0];
	}

	void prepareBuffers()
	{
		objectCount = OBJECT_COUNT * OBJECT_COUNT * OBJECT_COUNT;

		vks::Buffer stagingBuffer;

		std::vector<InstanceData> instanceData(objectCount);
		indirectCommands.resize(objectCount);

		// Indirect draw commands
		for (uint32_t x = 0; x < OBJECT_COUNT; x++)
		{
			for (uint32_t y = 0; y < OBJECT_COUNT; y++)
			{
				for (uint32_t z = 0; z < OBJECT_COUNT; z++)
				{
					uint32_t index = x + y * OBJECT_COUNT + z * OBJECT_COUNT * OBJECT_COUNT;
					indirectCommands[index].instanceCount = 1;
					indirectCommands[index].firstInstance = index;
					// firstIndex and indexCount are written by the compute shader
				}
			}
		}

		indirectStats.drawCount = static_cast<uint32_t>(indirectCommands.size());

		VK_CHECK_RESULT(vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eTransferSrc,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			&stagingBuffer,
			indirectCommands.size() * sizeof(vk::DrawIndexedIndirectCommand),
			indirectCommands.data()));

		VK_CHECK_RESULT(vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eIndirectBuffer | vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
			vk::MemoryPropertyFlagBits::eDeviceLocal,
			&indirectCommandsBuffer,
			stagingBuffer.size));

		vulkanDevice->copyBuffer(&stagingBuffer, &indirectCommandsBuffer, queue);

		stagingBuffer.destroy();

		VK_CHECK_RESULT(vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			&indirectDrawCountBuffer,
			sizeof(indirectStats)));

		// Map for host access
		indirectDrawCountBuffer.map();

		// Instance data
		for (uint32_t x = 0; x < OBJECT_COUNT; x++)
		{
			for (uint32_t y = 0; y < OBJECT_COUNT; y++)
			{
				for (uint32_t z = 0; z < OBJECT_COUNT; z++)
				{
					uint32_t index = x + y * OBJECT_COUNT + z * OBJECT_COUNT * OBJECT_COUNT;
					instanceData[index].pos = glm::vec3((float)x, (float)y, (float)z) - glm::vec3((float)OBJECT_COUNT / 2.0f);
					instanceData[index].scale = 2.0f;
				}
			}
		}

		VK_CHECK_RESULT(vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eTransferSrc,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			&stagingBuffer,
			instanceData.size() * sizeof(InstanceData),
			instanceData.data()));

		VK_CHECK_RESULT(vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
			vk::MemoryPropertyFlagBits::eDeviceLocal,
			&instanceBuffer,
			stagingBuffer.size));

		vulkanDevice->copyBuffer(&stagingBuffer, &instanceBuffer, queue);

		stagingBuffer.destroy();

		// Shader storage buffer containing index offsets and counts for the LODs
		struct LOD
		{
			uint32_t firstIndex;
			uint32_t indexCount;
			float distance;
			float _pad0;
		};
		std::vector<LOD> LODLevels;
		uint32_t n = 0;
		for (auto modelPart : models.lodObject.parts)
		{
			LOD lod;
			lod.firstIndex = modelPart.indexBase;			// First index for this LOD
			lod.indexCount = modelPart.indexCount;			// Index count for this LOD
			lod.distance = 5.0f + n * 5.0f;					// Starting distance (to viewer) for this LOD
			n++;
			LODLevels.push_back(lod);
		}

		VK_CHECK_RESULT(vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eTransferSrc,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			&stagingBuffer,
			LODLevels.size() * sizeof(LOD),
			LODLevels.data()));

		VK_CHECK_RESULT(vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
			vk::MemoryPropertyFlagBits::eDeviceLocal,
			&compute.lodLevelsBuffers,
			stagingBuffer.size));

		vulkanDevice->copyBuffer(&stagingBuffer, &compute.lodLevelsBuffers, queue);

		stagingBuffer.destroy();

		// Scene uniform buffer
		VK_CHECK_RESULT(vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eUniformBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			&uniformData.scene,
			sizeof(uboScene)));

		uniformData.scene.map();

		updateUniformBuffer(true);
	}

	void prepareCompute()
	{
		// Create a compute capable device queue
		//vk::DeviceQueueCreateInfo queueCreateInfo = {};

		//queueCreateInfo.pNext = NULL;
		//queueCreateInfo.queueFamilyIndex = vulkanDevice->queueFamilyIndices.compute;
		//queueCreateInfo.queueCount = 1;
		compute.queue = device.getQueue(vulkanDevice->queueFamilyIndices.compute);

		// Create compute pipeline
		// Compute pipelines are created separate from graphics pipelines even if they use the same queue (family index)

		std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings = {
			// Binding 0: Instance input data buffer
			vks::initializers::descriptorSetLayoutBinding(
				vk::DescriptorType::eStorageBuffer,
				vk::ShaderStageFlagBits::eCompute,
				0),
			// Binding 1: Indirect draw command output buffer (input)
			vks::initializers::descriptorSetLayoutBinding(
				vk::DescriptorType::eStorageBuffer,
				vk::ShaderStageFlagBits::eCompute,
				1),
			// Binding 2: Uniform buffer with global matrices (input)
			vks::initializers::descriptorSetLayoutBinding(
				vk::DescriptorType::eUniformBuffer,
				vk::ShaderStageFlagBits::eCompute,
				2),
			// Binding 3: Indirect draw stats (output)
			vks::initializers::descriptorSetLayoutBinding(
				vk::DescriptorType::eStorageBuffer,
				vk::ShaderStageFlagBits::eCompute,
				3),
			// Binding 4: LOD info (input)
			vks::initializers::descriptorSetLayoutBinding(
				vk::DescriptorType::eStorageBuffer,
				vk::ShaderStageFlagBits::eCompute,
				4),
		};

		vk::DescriptorSetLayoutCreateInfo descriptorLayout =
			vks::initializers::descriptorSetLayoutCreateInfo(
				setLayoutBindings.data(),
				static_cast<uint32_t>(setLayoutBindings.size()));

		compute.descriptorSetLayout = device.createDescriptorSetLayout(descriptorLayout);

		vk::PipelineLayoutCreateInfo pPipelineLayoutCreateInfo =
			vks::initializers::pipelineLayoutCreateInfo(
				&compute.descriptorSetLayout,
				1);

		compute.pipelineLayout = device.createPipelineLayout(pPipelineLayoutCreateInfo);

		vk::DescriptorSetAllocateInfo allocInfo =
			vks::initializers::descriptorSetAllocateInfo(
				descriptorPool,
				&compute.descriptorSetLayout,
				1);

		compute.descriptorSet = device.allocateDescriptorSets(allocInfo)[0];

		std::vector<vk::WriteDescriptorSet> computeWriteDescriptorSets =
		{
			// Binding 0: Instance input data buffer
			vks::initializers::writeDescriptorSet(
				compute.descriptorSet,
				vk::DescriptorType::eStorageBuffer,
				0,
				&instanceBuffer.descriptor),
			// Binding 1: Indirect draw command output buffer
			vks::initializers::writeDescriptorSet(
				compute.descriptorSet,
				vk::DescriptorType::eStorageBuffer,
				1,
				&indirectCommandsBuffer.descriptor),
			// Binding 2: Uniform buffer with global matrices
			vks::initializers::writeDescriptorSet(
				compute.descriptorSet,
				vk::DescriptorType::eUniformBuffer,
				2,
				&uniformData.scene.descriptor),
			// Binding 3: Atomic counter (written in shader)
			vks::initializers::writeDescriptorSet(
				compute.descriptorSet,
				vk::DescriptorType::eStorageBuffer,
				3,
				&indirectDrawCountBuffer.descriptor),
			// Binding 4: LOD info
			vks::initializers::writeDescriptorSet(
				compute.descriptorSet,
				vk::DescriptorType::eStorageBuffer,
				4,
				&compute.lodLevelsBuffers.descriptor)
		};

		device.updateDescriptorSets(computeWriteDescriptorSets);

		// Create pipeline		
		vk::ComputePipelineCreateInfo computePipelineCreateInfo = vks::initializers::computePipelineCreateInfo(compute.pipelineLayout, 0);
		computePipelineCreateInfo.stage = loadShader(getAssetPath() + "shaders/computecullandlod/cull.comp.spv", vk::ShaderStageFlagBits::eCompute);

		// Use specialization constants to pass max. level of detail (determined by no. of meshes)
		vk::SpecializationMapEntry specializationEntry{};
		specializationEntry.constantID = 0;
		specializationEntry.offset = 0;
		specializationEntry.size = sizeof(uint32_t);

		uint32_t specializationData = static_cast<uint32_t>(models.lodObject.parts.size()) - 1;

		vk::SpecializationInfo specializationInfo;
		specializationInfo.mapEntryCount = 1;
		specializationInfo.pMapEntries = &specializationEntry;
		specializationInfo.dataSize = sizeof(specializationData);
		specializationInfo.pData = &specializationData;

		computePipelineCreateInfo.stage.pSpecializationInfo = &specializationInfo;

		compute.pipeline = device.createComputePipelines(pipelineCache, computePipelineCreateInfo);

		// Separate command pool as queue family for compute may be different than graphics
		vk::CommandPoolCreateInfo cmdPoolInfo = {};

		cmdPoolInfo.queueFamilyIndex = vulkanDevice->queueFamilyIndices.compute;
		cmdPoolInfo.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
		compute.commandPool = device.createCommandPool(cmdPoolInfo);

		// Create a command buffer for compute operations
		vk::CommandBufferAllocateInfo cmdBufAllocateInfo =
			vks::initializers::commandBufferAllocateInfo(
				compute.commandPool,
				vk::CommandBufferLevel::ePrimary,
				1);

		compute.commandBuffer) = device.allocateCommandBuffers(cmdBufAllocateInfo);

		// Fence for compute CB sync
		vk::FenceCreateInfo fenceCreateInfo = vks::initializers::fenceCreateInfo(VK_FENCE_CREATE_SIGNALED_BIT);
		compute.fence = device.createFence(fenceCreateInfo);

		vk::SemaphoreCreateInfo semaphoreCreateInfo = vks::initializers::semaphoreCreateInfo();
		compute.semaphore = device.createSemaphore(semaphoreCreateInfo);
		
		// Build a single command buffer containing the compute dispatch commands
		buildComputeCommandBuffer();
	}

	void updateUniformBuffer(bool viewChanged)
	{
		if (viewChanged)
		{
			uboScene.projection = camera.matrices.perspective;
			uboScene.modelview = camera.matrices.view;
			if (!fixedFrustum)
			{
				uboScene.cameraPos = glm::vec4(camera.position, 1.0f) * -1.0f;
				frustum.update(uboScene.projection * uboScene.modelview);
				memcpy(uboScene.frustumPlanes, frustum.planes.data(), sizeof(glm::vec4) * 6);
			}
		}

		memcpy(uniformData.scene.mapped, &uboScene, sizeof(uboScene));
	}

	void draw()
	{
		VulkanExampleBase::prepareFrame();

		// Submit compute shader for frustum culling

		// Wait for fence to ensure that compute buffer writes have finished
		vkWaitForFences(device, 1, &compute.fence, VK_TRUE, UINT64_MAX);
		device.resetFences(compute.fence);

		vk::SubmitInfo computeSubmitInfo = vks::initializers::submitInfo();
		computeSubmitInfo.commandBufferCount = 1;
		computeSubmitInfo.pCommandBuffers = &compute.commandBuffer;
		computeSubmitInfo.signalSemaphoreCount = 1;
		computeSubmitInfo.pSignalSemaphores = &compute.semaphore;

		compute.queue.submit(computeSubmitInfo, vk::Fence(nullptr));
		
		// Submit graphics command buffer

		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &drawCmdBuffers[currentBuffer];

		// Wait on present and compute semaphores
		std::array<vk::PipelineStageFlags,2> stageFlags = {
			vk::PipelineStageFlagBits::eColorAttachmentOutput,
			vk::PipelineStageFlagBits::eComputeShader,
		};
		std::array<vk::Semaphore,2> waitSemaphores = {
			semaphores.presentComplete,						// Wait for presentation to finished
			compute.semaphore								// Wait for compute to finish
		};

		submitInfo.pWaitSemaphores = waitSemaphores.data();
		submitInfo.waitSemaphoreCount = static_cast<uint32_t>(waitSemaphores.size());
		submitInfo.pWaitDstStageMask = stageFlags.data();

		// Submit to queue
		queue.submit(submitInfo, compute.fence);

		VulkanExampleBase::submitFrame();

		// Get draw count from compute
		memcpy(&indirectStats, indirectDrawCountBuffer.mapped, sizeof(indirectStats));
	}

	void prepare()
	{
		VulkanExampleBase::prepare();
		loadAssets();
		setupVertexDescriptions();
		prepareBuffers();
		setupDescriptorSetLayout();
		preparePipelines();
		setupDescriptorPool();
		setupDescriptorSet();
		prepareCompute();
		buildCommandBuffers();
		prepared = true;
	}

	virtual void render()
	{
		if (!prepared)
		{
			return;
		}
		draw();
	}

	virtual void viewChanged()
	{
		updateUniformBuffer(true);
	}

	virtual void keyPressed(uint32_t keyCode)
	{
		switch (keyCode)
		{
		case KEY_F:
		case GAMEPAD_BUTTON_A:
			fixedFrustum = !fixedFrustum;
			updateUniformBuffer(true);
			break;
		}
	}

	virtual void getOverlayText(VulkanTextOverlay *textOverlay)
	{
#if defined(__ANDROID__)
		textOverlay->addText("\"Button A\" to freeze frustum", 5.0f, 85.0f, VulkanTextOverlay::alignLeft);
#else
		textOverlay->addText("\"f\" to freeze frustum", 5.0f, 85.0f, VulkanTextOverlay::alignLeft);
#endif
		textOverlay->addText("visible: " + std::to_string(indirectStats.drawCount), 5.0f, 110.0f, VulkanTextOverlay::alignLeft);
		for (uint32_t i = 0; i < MAX_LOD_LEVEL + 1; i++)
		{
			textOverlay->addText("lod " + std::to_string(i) + ": " + std::to_string(indirectStats.lodCount[i]), 5.0f, 125.0f + (float)i * 20.0f, VulkanTextOverlay::alignLeft);
		}
	}
};

VULKAN_EXAMPLE_MAIN()