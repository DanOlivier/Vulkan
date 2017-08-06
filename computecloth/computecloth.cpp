/*
* Vulkan Example - Compute shader sloth simulation
*
* Updated compute shader by Lukas Bergdoll (https://github.com/Voultapher)
*
* Copyright (C) 2016-2017 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <vector>
#include <random>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <vulkan/vulkan.hpp>
#include "vulkanexamplebase.h"
#include "VulkanTexture.hpp"
#include "VulkanModel.hpp"

#define ENABLE_VALIDATION false

class VulkanExample : public VulkanExampleBase
{
public:
	uint32_t sceneSetup = 0;
	uint32_t readSet = 0;
	uint32_t indexCount;
	bool simulateWind = false;

	vks::Texture2D textureCloth;

	vks::VertexLayout vertexLayout = vks::VertexLayout({
		vks::VERTEX_COMPONENT_POSITION,
		vks::VERTEX_COMPONENT_UV,
		vks::VERTEX_COMPONENT_NORMAL,
	});
	vks::Model modelSphere;

	// Resources for the graphics part of the example
	struct {
		vk::DescriptorSetLayout descriptorSetLayout;
		vk::DescriptorSet descriptorSet;
		vk::PipelineLayout pipelineLayout;
		struct Pipelines {
			vk::Pipeline cloth;
			vk::Pipeline sphere;
		} pipelines;
		vks::Buffer indices;
		vks::Buffer uniformBuffer;
		struct graphicsUBO {
			glm::mat4 projection;
			glm::mat4 view;
			glm::vec4 lightPos = glm::vec4(-1.0f, 2.0f, -1.0f, 1.0f);
		} ubo;
	} graphics;

	// Resources for the compute part of the example
	struct {
		struct StorageBuffers {
			vks::Buffer input;
			vks::Buffer output;
		} storageBuffers;
		vks::Buffer uniformBuffer;
		vk::Queue queue;
		vk::CommandPool commandPool;
		std::array<vk::CommandBuffer,2> commandBuffers;
		vk::Fence fence;
		vk::DescriptorSetLayout descriptorSetLayout;
		std::array<vk::DescriptorSet,2> descriptorSets;
		vk::PipelineLayout pipelineLayout;
		vk::Pipeline pipeline;
		struct computeUBO {
			float deltaT = 0.0f;
			float particleMass = 0.1f;
			float springStiffness = 2000.0f;
			float damping = 0.25f;
			float restDistH;
			float restDistV;
			float restDistD;
			float sphereRadius = 0.5f;
			glm::vec4 spherePos = glm::vec4(0.0f, 0.0f, 0.0f, 0.0f);
			glm::vec4 gravity = glm::vec4(0.0f, 9.8f, 0.0f, 0.0f);
			glm::ivec2 particleCount;
		} ubo;
	} compute;

	// SSBO cloth grid particle declaration
	struct Particle {
		glm::vec4 pos;
		glm::vec4 vel;
		glm::vec4 uv;
		glm::vec4 normal;
		float pinned;
		glm::vec3 _pad0;
	};

	struct Cloth {
		glm::uvec2 gridsize = glm::uvec2(60, 60);
		glm::vec2 size = glm::vec2(2.5f, 2.5f);
	} cloth;

	VulkanExample() : VulkanExampleBase(ENABLE_VALIDATION)
	{
		enableTextOverlay = true;
		title = "Vulkan Example - Compute shader cloth simulation";
		camera.type = Camera::CameraType::lookat;
		camera.setPerspective(60.0f, (float)width / (float)height, 0.1f, 512.0f);
		camera.setRotation(glm::vec3(-30.0f, -45.0f, 0.0f));
		camera.setTranslation(glm::vec3(0.0f, 0.0f, -3.5f));
		srand((unsigned int)time(NULL));
	}

	~VulkanExample()
	{
		// Graphics
		graphics.uniformBuffer.destroy();
		device.destroyPipeline(graphics.pipelines.cloth);
		device.destroyPipeline(graphics.pipelines.sphere);
		device.destroyPipelineLayout(graphics.pipelineLayout);
		device.destroyDescriptorSetLayout(graphics.descriptorSetLayout);
		textureCloth.destroy();
		modelSphere.destroy();

		// Compute
		compute.storageBuffers.input.destroy();
		compute.storageBuffers.output.destroy();
		compute.uniformBuffer.destroy();
		device.destroyPipelineLayout(compute.pipelineLayout);
		device.destroyDescriptorSetLayout(compute.descriptorSetLayout);
		device.destroyPipeline(compute.pipeline);
		device.destroyFence(compute.fence);
		device.destroyCommandPool(compute.commandPool);
	}

	// Enable physical device features required for this example				
	virtual void getEnabledFeatures()
	{
		if (deviceFeatures.samplerAnisotropy) {
			enabledFeatures.samplerAnisotropy = VK_TRUE;
		}
	};

	void loadAssets()
	{
		textureCloth.loadFromFile(getAssetPath() + "textures/vulkan_cloth_rgba.ktx", vk::Format::eR8G8B8A8Unorm, vulkanDevice, queue);
		modelSphere.loadFromFile(ASSET_PATH "models/geosphere.obj", vertexLayout, compute.ubo.sphereRadius * 0.05f, vulkanDevice, queue);
	}

	void buildCommandBuffers()
	{
		// Destroy command buffers if already present
		if (!checkCommandBuffers())
		{
			destroyCommandBuffers();
			createCommandBuffers();
		}

		vk::CommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();

		vk::ClearValue clearValues[2];
		clearValues[0].color = vk::ClearColorValue{ std::array<float, 4>{ 0.0f, 0.0f, 0.0f, 1.0f } };;
		clearValues[1].depthStencil = vk::ClearDepthStencilValue{ 1.0f, 0 };

		vk::RenderPassBeginInfo renderPassBeginInfo = vks::initializers::renderPassBeginInfo();
		renderPassBeginInfo.renderPass = renderPass;
		renderPassBeginInfo.renderArea.offset.x = 0;
		renderPassBeginInfo.renderArea.offset.y = 0;
		renderPassBeginInfo.renderArea.extent.width = width;
		renderPassBeginInfo.renderArea.extent.height = height;
		renderPassBeginInfo.clearValueCount = 2;
		renderPassBeginInfo.pClearValues = clearValues;

		for (int32_t i = 0; i < drawCmdBuffers.size(); ++i)
		{
			// Set target frame buffer
			renderPassBeginInfo.framebuffer = frameBuffers[i];

			drawCmdBuffers[i].begin(cmdBufInfo);

			// Draw the particle system using the update vertex buffer

			drawCmdBuffers[i].beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);

			vk::Viewport viewport = vks::initializers::viewport((float)width, (float)height, 0.0f, 1.0f);
			drawCmdBuffers[i].setViewport(0, viewport);

			vk::Rect2D scissor = vks::initializers::rect2D(width, height, 0, 0);
			drawCmdBuffers[i].setScissor(0, scissor);

			std::vector<vk::DeviceSize> offsets = { 0 };

			// Render sphere
			if (sceneSetup == 0) {
				drawCmdBuffers[i].bindPipeline(vk::PipelineBindPoint::eGraphics, graphics.pipelines.sphere);
				drawCmdBuffers[i].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, graphics.pipelineLayout, 0, graphics.descriptorSet, nullptr);
				drawCmdBuffers[i].bindIndexBuffer(modelSphere.indices.buffer, 0, vk::IndexType::eUint32);
				drawCmdBuffers[i].bindVertexBuffers(0, modelSphere.vertices.buffer, offsets);
				drawCmdBuffers[i].drawIndexed(modelSphere.indexCount, 1, 0, 0, 0);
			}

			// Render cloth
			drawCmdBuffers[i].bindPipeline(vk::PipelineBindPoint::eGraphics, graphics.pipelines.cloth);
			drawCmdBuffers[i].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, graphics.pipelineLayout, 0, graphics.descriptorSet, nullptr);
			drawCmdBuffers[i].bindIndexBuffer(graphics.indices.buffer, 0, vk::IndexType::eUint32);
			drawCmdBuffers[i].bindVertexBuffers(0, compute.storageBuffers.output.buffer, offsets);
			drawCmdBuffers[i].drawIndexed(indexCount, 1, 0, 0, 0);

			drawCmdBuffers[i].endRenderPass();

			drawCmdBuffers[i].end();
		}

	}

	// todo: check barriers (validation, separate compute queue)
	void buildComputeCommandBuffer()
	{
		vk::CommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();

		for (uint32_t i = 0; i < 2; i++) {

			compute.commandBuffers[i].begin(cmdBufInfo);

			vk::BufferMemoryBarrier bufferBarrier = vks::initializers::bufferMemoryBarrier();
			bufferBarrier.srcAccessMask = vk::AccessFlagBits::eVertexAttributeRead;
			bufferBarrier.dstAccessMask = vk::AccessFlagBits::eShaderWrite;
			bufferBarrier.srcQueueFamilyIndex = vulkanDevice->queueFamilyIndices.graphics;
			bufferBarrier.dstQueueFamilyIndex = vulkanDevice->queueFamilyIndices.compute;
			bufferBarrier.size = VK_WHOLE_SIZE;

			std::vector<vk::BufferMemoryBarrier> bufferBarriers;
			bufferBarrier.buffer = compute.storageBuffers.input.buffer;
			bufferBarriers.push_back(bufferBarrier);
			bufferBarrier.buffer = compute.storageBuffers.output.buffer;
			bufferBarriers.push_back(bufferBarrier);

			compute.commandBuffers[i].pipelineBarrier(
				vk::PipelineStageFlagBits::eComputeShader,
				vk::PipelineStageFlagBits::eComputeShader,
				vk::DependencyFlags(),
				 nullptr,
				bufferBarriers,
				 nullptr);

			compute.commandBuffers[i].bindPipeline(vk::PipelineBindPoint::eCompute, compute.pipeline);

			uint32_t calculateNormals = 0;
			compute.commandBuffers[i].pushConstants(compute.pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, calculateNormals);

			// Dispatch the compute job
			const uint32_t iterations = 64;
			for (uint32_t j = 0; j < iterations; j++) {
				readSet = 1 - readSet;
				compute.commandBuffers[i].bindDescriptorSets(vk::PipelineBindPoint::eCompute, compute.pipelineLayout, 0, compute.descriptorSets[readSet], nullptr);

				if (j == iterations - 1) {
					calculateNormals = 1;
					compute.commandBuffers[i].pushConstants(compute.pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, calculateNormals);
				}

				vkCmdDispatch(compute.commandBuffers[i], cloth.gridsize.x / 10, cloth.gridsize.y / 10, 1);

				for (auto &barrier : bufferBarriers) {
					barrier.srcAccessMask = vk::AccessFlagBits::eShaderWrite;
					barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;
					barrier.srcQueueFamilyIndex = vulkanDevice->queueFamilyIndices.compute;
					barrier.dstQueueFamilyIndex = vulkanDevice->queueFamilyIndices.graphics;
				}

				compute.commandBuffers[i].pipelineBarrier(
					vk::PipelineStageFlagBits::eComputeShader,
					vk::PipelineStageFlagBits::eComputeShader,
					vk::DependencyFlags(),
					nullptr,
					bufferBarriers,
					nullptr);

			}

			for (auto &barrier : bufferBarriers) {
				barrier.srcAccessMask = vk::AccessFlagBits::eShaderWrite;
				barrier.dstAccessMask = vk::AccessFlagBits::eVertexAttributeRead;
				barrier.srcQueueFamilyIndex = vulkanDevice->queueFamilyIndices.compute;
				barrier.dstQueueFamilyIndex = vulkanDevice->queueFamilyIndices.graphics;
			}

			compute.commandBuffers[i].pipelineBarrier(
				compute.commandBuffers[i],
				vk::PipelineStageFlagBits::eComputeShader,
				vk::PipelineStageFlagBits::eComputeShader,
				vk::DependencyFlags(),
				nullptr,
				bufferBarriers,
				nullptr);

			vkEndCommandBuffer(compute.commandBuffers[i]);
		}
	}

	// Setup and fill the compute shader storage buffers containing the particles
	void prepareStorageBuffers()
	{
		std::vector<Particle> particleBuffer(cloth.gridsize.x *  cloth.gridsize.y);

		float dx =  cloth.size.x / (cloth.gridsize.x - 1);
		float dy =  cloth.size.y / (cloth.gridsize.y - 1);
		float du = 1.0f / (cloth.gridsize.x - 1);
		float dv = 1.0f / (cloth.gridsize.y - 1);

		switch (sceneSetup) {
			case 0 :
			{
				// Horz. cloth falls onto sphere
				glm::mat4 transM = glm::translate(glm::mat4(), glm::vec3(- cloth.size.x / 2.0f, -2.0f, - cloth.size.y / 2.0f));
				for (uint32_t i = 0; i <  cloth.gridsize.y; i++) {
					for (uint32_t j = 0; j <  cloth.gridsize.x; j++) {
						particleBuffer[i + j * cloth.gridsize.y].pos = transM * glm::vec4(dx * j, 0.0f, dy * i, 1.0f);
						particleBuffer[i + j * cloth.gridsize.y].vel = glm::vec4(0.0f);
						particleBuffer[i + j * cloth.gridsize.y].uv = glm::vec4(1.0f - du * i, dv * j, 0.0f, 0.0f);
					}
				}
				break;
			}
			case 1:
			{
				// Vert. Pinned cloth
				glm::mat4 transM = glm::translate(glm::mat4(), glm::vec3(- cloth.size.x / 2.0f, - cloth.size.y / 2.0f, 0.0f));
				for (uint32_t i = 0; i <  cloth.gridsize.y; i++) {
					for (uint32_t j = 0; j <  cloth.gridsize.x; j++) {
						particleBuffer[i + j * cloth.gridsize.y].pos = transM * glm::vec4(dx * j, dy * i, 0.0f, 1.0f);
						particleBuffer[i + j * cloth.gridsize.y].vel = glm::vec4(0.0f);
						particleBuffer[i + j * cloth.gridsize.y].uv = glm::vec4(du * j, dv * i, 0.0f, 0.0f);
						// Pin some particles
						particleBuffer[i + j * cloth.gridsize.y].pinned = (i == 0) && ((j == 0) || (j ==  cloth.gridsize.x / 3) || (j ==  cloth.gridsize.x -  cloth.gridsize.x / 3) || (j ==  cloth.gridsize.x - 1));
						// Remove sphere
						compute.ubo.spherePos.z = -10.0f;
					}
				}
				break;
			}
		}

		vk::DeviceSize storageBufferSize = particleBuffer.size() * sizeof(Particle);

		// Staging
		// SSBO won't be changed on the host after upload so copy to device local memory 

		vks::Buffer stagingBuffer;

		vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eTransferSrc,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			&stagingBuffer,
			storageBufferSize,
			particleBuffer.data());

		vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
			vk::MemoryPropertyFlagBits::eDeviceLocal,
			&compute.storageBuffers.input,
			storageBufferSize);

		vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
			vk::MemoryPropertyFlagBits::eDeviceLocal,
			&compute.storageBuffers.output,
			storageBufferSize);

		// Copy from staging buffer
		vk::CommandBuffer copyCmd = VulkanExampleBase::createCommandBuffer(vk::CommandBufferLevel::ePrimary, true);
		vk::BufferCopy copyRegion = {};
		copyRegion.size = storageBufferSize;
		copyCmd.copyBuffer(stagingBuffer.buffer, compute.storageBuffers.input.buffer, copyRegion);
		copyCmd.copyBuffer(stagingBuffer.buffer, compute.storageBuffers.output.buffer, copyRegion);
		VulkanExampleBase::flushCommandBuffer(copyCmd, queue, true);

		stagingBuffer.destroy();

		// Indices
		std::vector<uint32_t> indices;
		for (uint32_t y = 0; y <  cloth.gridsize.y - 1; y++) {
			for (uint32_t x = 0; x <  cloth.gridsize.x; x++) {
				indices.push_back((y + 1) *  cloth.gridsize.x + x);
				indices.push_back((y)*  cloth.gridsize.x + x);
			}
			// Primitive restart (signlaed by special value 0xFFFFFFFF)
			indices.push_back(0xFFFFFFFF);
		}
		uint32_t indexBufferSize = static_cast<uint32_t>(indices.size()) * sizeof(uint32_t);
		indexCount = static_cast<uint32_t>(indices.size());

		vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eTransferSrc,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			&stagingBuffer,
			indexBufferSize,
			indices.data());

		vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst,
			vk::MemoryPropertyFlagBits::eDeviceLocal,
			&graphics.indices,
			indexBufferSize);

		// Copy from staging buffer
		copyCmd = VulkanExampleBase::createCommandBuffer(vk::CommandBufferLevel::ePrimary, true);
		copyRegion = {};
		copyRegion.size = indexBufferSize;
		copyCmd.copyBuffer(stagingBuffer.buffer, graphics.indices.buffer, copyRegion);
		VulkanExampleBase::flushCommandBuffer(copyCmd, queue, true);

		stagingBuffer.destroy();
	}

	void setupDescriptorPool()
	{
		std::vector<vk::DescriptorPoolSize> poolSizes = {
			vks::initializers::descriptorPoolSize(vk::DescriptorType::eUniformBuffer, 3),
			vks::initializers::descriptorPoolSize(vk::DescriptorType::eStorageBuffer, 4),
			vks::initializers::descriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, 2)
		};

		vk::DescriptorPoolCreateInfo descriptorPoolInfo =
			vks::initializers::descriptorPoolCreateInfo(poolSizes, 3);

		descriptorPool = device.createDescriptorPool(descriptorPoolInfo);
	}

	void setupLayoutsAndDescriptors()
	{
		// Set layout
		std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings = {
			vks::initializers::descriptorSetLayoutBinding(vk::DescriptorType::eUniformBuffer, vk::ShaderStageFlagBits::eVertex, 0),
			vks::initializers::descriptorSetLayoutBinding(vk::DescriptorType::eCombinedImageSampler, vk::ShaderStageFlagBits::eFragment, 1)
		};
		vk::DescriptorSetLayoutCreateInfo descriptorLayout =
			vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
		graphics.descriptorSetLayout = device.createDescriptorSetLayout(descriptorLayout);

		vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo =
			vks::initializers::pipelineLayoutCreateInfo(&graphics.descriptorSetLayout, 1);
		graphics.pipelineLayout = device.createPipelineLayout(pipelineLayoutCreateInfo);

		// Set
		vk::DescriptorSetAllocateInfo allocInfo = 
			vks::initializers::descriptorSetAllocateInfo(descriptorPool, &graphics.descriptorSetLayout, 1);
		graphics.descriptorSet = device.allocateDescriptorSets(allocInfo)[0];

		std::vector<vk::WriteDescriptorSet> writeDescriptorSets = {
			vks::initializers::writeDescriptorSet(graphics.descriptorSet, vk::DescriptorType::eUniformBuffer, 0, &graphics.uniformBuffer.descriptor),
			vks::initializers::writeDescriptorSet(graphics.descriptorSet, vk::DescriptorType::eCombinedImageSampler, 1, &textureCloth.descriptor)
		};
		device.updateDescriptorSets(writeDescriptorSets, nullptr);
	}

	void preparePipelines()
	{
		vk::PipelineInputAssemblyStateCreateInfo inputAssemblyState =
			vks::initializers::pipelineInputAssemblyStateCreateInfo(vk::PrimitiveTopology::eTriangleStrip, vk::PipelineInputAssemblyStateCreateFlags(), VK_TRUE);

		vk::PipelineRasterizationStateCreateInfo rasterizationState =
			vks::initializers::pipelineRasterizationStateCreateInfo(vk::PolygonMode::eFill, vk::CullModeFlagBits::eNone, vk::FrontFace::eCounterClockwise);

		vk::PipelineColorBlendAttachmentState blendAttachmentState =
			vks::initializers::pipelineColorBlendAttachmentState(
				vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA, 
				VK_FALSE);

		vk::PipelineColorBlendStateCreateInfo colorBlendState =
			vks::initializers::pipelineColorBlendStateCreateInfo(1, &blendAttachmentState);

		vk::PipelineDepthStencilStateCreateInfo depthStencilState =
			vks::initializers::pipelineDepthStencilStateCreateInfo(VK_TRUE, VK_TRUE, vk::CompareOp::eLessOrEqual);

		vk::PipelineViewportStateCreateInfo viewportState =
			vks::initializers::pipelineViewportStateCreateInfo(1, 1);

		vk::PipelineMultisampleStateCreateInfo multisampleState =
			vks::initializers::pipelineMultisampleStateCreateInfo(vk::SampleCountFlagBits::e1);

		std::vector<vk::DynamicState> dynamicStateEnables = {
			vk::DynamicState::eViewport,
			vk::DynamicState::eScissor
		};
		vk::PipelineDynamicStateCreateInfo dynamicState =
			vks::initializers::pipelineDynamicStateCreateInfo(dynamicStateEnables);

		// Rendering pipeline
		std::array<vk::PipelineShaderStageCreateInfo,2> shaderStages;

		shaderStages[0] = loadShader(getAssetPath() + "shaders/computecloth/cloth.vert.spv", vk::ShaderStageFlagBits::eVertex);
		shaderStages[1] = loadShader(getAssetPath() + "shaders/computecloth/cloth.frag.spv", vk::ShaderStageFlagBits::eFragment);

		vk::GraphicsPipelineCreateInfo pipelineCreateInfo =
			vks::initializers::pipelineCreateInfo(
				graphics.pipelineLayout,
				renderPass);

		// Input attributes	

		// Binding description
		std::vector<vk::VertexInputBindingDescription> inputBindings = {
			vks::initializers::vertexInputBindingDescription(0, sizeof(Particle), vk::VertexInputRate::eVertex)
		};

		// Attribute descriptions
		std::vector<vk::VertexInputAttributeDescription> inputAttributes = {
			vks::initializers::vertexInputAttributeDescription(0, 0, vk::Format::eR32G32B32Sfloat, offsetof(Particle, pos)),
			vks::initializers::vertexInputAttributeDescription(0, 1, vk::Format::eR32G32Sfloat, offsetof(Particle, uv)),
			vks::initializers::vertexInputAttributeDescription(0, 2, vk::Format::eR32G32B32Sfloat, offsetof(Particle, normal))
		};

		// Assign to vertex buffer
		vk::PipelineVertexInputStateCreateInfo inputState = vks::initializers::pipelineVertexInputStateCreateInfo();
		inputState.vertexBindingDescriptionCount = static_cast<uint32_t>(inputBindings.size());
		inputState.pVertexBindingDescriptions = inputBindings.data();
		inputState.vertexAttributeDescriptionCount = static_cast<uint32_t>(inputAttributes.size());
		inputState.pVertexAttributeDescriptions = inputAttributes.data();

		pipelineCreateInfo.pVertexInputState = &inputState;
		pipelineCreateInfo.pInputAssemblyState = &inputAssemblyState;
		pipelineCreateInfo.pRasterizationState = &rasterizationState;
		pipelineCreateInfo.pColorBlendState = &colorBlendState;
		pipelineCreateInfo.pMultisampleState = &multisampleState;
		pipelineCreateInfo.pViewportState = &viewportState;
		pipelineCreateInfo.pDepthStencilState = &depthStencilState;
		pipelineCreateInfo.pDynamicState = &dynamicState;
		pipelineCreateInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
		pipelineCreateInfo.pStages = shaderStages.data();
		pipelineCreateInfo.renderPass = renderPass;

		graphics.pipelines.cloth = device.createGraphicsPipelines(pipelineCache, pipelineCreateInfo)[0];

		// Sphere rendering pipeline
		inputBindings = {
			vks::initializers::vertexInputBindingDescription(0, vertexLayout.stride(), vk::VertexInputRate::eVertex)
		};
		inputAttributes = {
			vks::initializers::vertexInputAttributeDescription(0, 0, vk::Format::eR32G32B32Sfloat, 0),
			vks::initializers::vertexInputAttributeDescription(0, 1, vk::Format::eR32G32Sfloat, sizeof(float) * 3),
			vks::initializers::vertexInputAttributeDescription(0, 2, vk::Format::eR32G32B32Sfloat, sizeof(float) * 5)
		};
		inputState.vertexAttributeDescriptionCount = static_cast<uint32_t>(inputAttributes.size());
		inputAssemblyState.topology = vk::PrimitiveTopology::eTriangleList;
		rasterizationState.polygonMode = vk::PolygonMode::eFill;
		shaderStages[0] = loadShader(getAssetPath() + "shaders/computecloth/sphere.vert.spv", vk::ShaderStageFlagBits::eVertex);
		shaderStages[1] = loadShader(getAssetPath() + "shaders/computecloth/sphere.frag.spv", vk::ShaderStageFlagBits::eFragment);
		graphics.pipelines.sphere = device.createGraphicsPipelines(pipelineCache, pipelineCreateInfo)[0];
	}

	void prepareCompute()
	{
		// Create a compute capable device queue
		compute.queue = device.getQueue(vulkanDevice->queueFamilyIndices.compute);

		// Create compute pipeline
		std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings = {
			vks::initializers::descriptorSetLayoutBinding(vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute, 0),
			vks::initializers::descriptorSetLayoutBinding(vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute, 1),
			vks::initializers::descriptorSetLayoutBinding(vk::DescriptorType::eUniformBuffer, vk::ShaderStageFlagBits::eCompute, 2),
		};

		vk::DescriptorSetLayoutCreateInfo descriptorLayout =
			vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);

		compute.descriptorSetLayout = device.createDescriptorSetLayout(descriptorLayout));

		vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo =
			vks::initializers::pipelineLayoutCreateInfo(&compute.descriptorSetLayout, 1);

		// Push constants used to pass some parameters
		vk::PushConstantRange pushConstantRange =
			vks::initializers::pushConstantRange(vk::ShaderStageFlagBits::eCompute, sizeof(uint32_t), 0);
		pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
		pipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRange;

		VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr,	&compute.pipelineLayout));

		vk::DescriptorSetAllocateInfo allocInfo =
			vks::initializers::descriptorSetAllocateInfo(descriptorPool, &compute.descriptorSetLayout, 1);

		// Create two descriptor sets with input and output buffers switched 
		compute.descriptorSets[0] = device.allocateDescriptorSets(allocInfo)[0];
		compute.descriptorSets[1] = device.allocateDescriptorSets(allocInfo)[0];

		std::vector<vk::WriteDescriptorSet> computeWriteDescriptorSets = {
			vks::initializers::writeDescriptorSet(compute.descriptorSets[0], vk::DescriptorType::eStorageBuffer, 0, &compute.storageBuffers.input.descriptor),
			vks::initializers::writeDescriptorSet(compute.descriptorSets[0], vk::DescriptorType::eStorageBuffer, 1, &compute.storageBuffers.output.descriptor),
			vks::initializers::writeDescriptorSet(compute.descriptorSets[0], vk::DescriptorType::eUniformBuffer, 2, &compute.uniformBuffer.descriptor),

			vks::initializers::writeDescriptorSet(compute.descriptorSets[1], vk::DescriptorType::eStorageBuffer, 0, &compute.storageBuffers.output.descriptor),
			vks::initializers::writeDescriptorSet(compute.descriptorSets[1], vk::DescriptorType::eStorageBuffer, 1, &compute.storageBuffers.input.descriptor),
			vks::initializers::writeDescriptorSet(compute.descriptorSets[1], vk::DescriptorType::eUniformBuffer, 2, &compute.uniformBuffer.descriptor)
		};

		device.updateDescriptorSets(writeDescriptorSets, nullptr)(computeWriteDescriptorSets);

		// Create pipeline		
		vk::ComputePipelineCreateInfo computePipelineCreateInfo = vks::initializers::computePipelineCreateInfo(compute.pipelineLayout);
		computePipelineCreateInfo.stage = loadShader(getAssetPath() + "shaders/computecloth/cloth.comp.spv", vk::ShaderStageFlagBits::eCompute);
		compute.pipeline = device.createComputePipelines(pipelineCache, computePipelineCreateInfo)[0];

		// Separate command pool as queue family for compute may be different than graphics
		vk::CommandPoolCreateInfo cmdPoolInfo = {};

		cmdPoolInfo.queueFamilyIndex = vulkanDevice->queueFamilyIndices.compute;
		cmdPoolInfo.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
		compute.commandPool = device.createCommandPool(cmdPoolInfo);

		// Create a command buffer for compute operations
		vk::CommandBufferAllocateInfo cmdBufAllocateInfo =
			vks::initializers::commandBufferAllocateInfo(compute.commandPool, vk::CommandBufferLevel::ePrimary, 2);	

		compute.commandBuffers = device.allocateCommandBuffers(cmdBufAllocateInfo)[0];

		// Fence for compute CB sync
		vk::FenceCreateInfo fenceCreateInfo = vks::initializers::fenceCreateInfo(vk::FenceCreateFlagBits::eSignaled);
		compute.fence = device.createFence(fenceCreateInfo);

		// Build a single command buffer containing the compute dispatch commands
		buildComputeCommandBuffer();
	}

	// Prepare and initialize uniform buffer containing shader uniforms
	void prepareUniformBuffers()
	{
		// Compute shader uniform buffer block
		vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eUniformBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			&compute.uniformBuffer,
			sizeof(compute.ubo));
		compute.uniformBuffer.map();
	
		// Initial values
		float dx = cloth.size.x / (cloth.gridsize.x - 1);
		float dy = cloth.size.y / (cloth.gridsize.y - 1);

		compute.ubo.restDistH = dx;
		compute.ubo.restDistV = dy;
		compute.ubo.restDistD = sqrtf(dx * dx + dy * dy);
		compute.ubo.particleCount = cloth.gridsize;

		updateComputeUBO();

		// Vertex shader uniform buffer block
		vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eUniformBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			&graphics.uniformBuffer,
			sizeof(graphics.ubo));
		graphics.uniformBuffer.map();

		updateGraphicsUBO();
	}

	void updateComputeUBO()
	{
		if (!paused) {
			compute.ubo.deltaT = 0.000005f;
			// todo: base on frametime
			//compute.ubo.deltaT = frameTimer * 0.0075f;

			std::mt19937 rg((unsigned)time(nullptr));
			std::uniform_real_distribution<float> rd(1.0f, 6.0f);

			if (simulateWind) {
				compute.ubo.gravity.x = cos(glm::radians(-timer * 360.0f)) * (rd(rg) - rd(rg));
				compute.ubo.gravity.z = sin(glm::radians(timer * 360.0f)) * (rd(rg) - rd(rg));
			}
			else {
				compute.ubo.gravity.x = 0.0f;
				compute.ubo.gravity.z = 0.0f;
			}
		}
		else {
			compute.ubo.deltaT = 0.0f;
		}
		memcpy(compute.uniformBuffer.mapped, &compute.ubo, sizeof(compute.ubo));
	}

	void updateGraphicsUBO()
	{
		graphics.ubo.projection = camera.matrices.perspective;
		graphics.ubo.view = camera.matrices.view;
		memcpy(graphics.uniformBuffer.mapped, &graphics.ubo, sizeof(graphics.ubo));
	}

	void draw()
	{
		// Submit graphics commands
		VulkanExampleBase::prepareFrame();

		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &drawCmdBuffers[currentBuffer];
		queue.submit(submitInfo, vk::Fence(nullptr));

		VulkanExampleBase::submitFrame();

		device.waitForFences(compute.fence, VK_TRUE, UINT64_MAX);
		device.resetFences(compute.fence);

		vk::SubmitInfo computeSubmitInfo = vks::initializers::submitInfo();
		computeSubmitInfo.commandBufferCount = 1;
		computeSubmitInfo.pCommandBuffers = &compute.commandBuffers[readSet];

		compute.queue.submit(computeSubmitInfo, compute.fence);
	}

	void prepare()
	{
		VulkanExampleBase::prepare();
		loadAssets();
		prepareStorageBuffers();
		prepareUniformBuffers();
		setupDescriptorPool();
		setupLayoutsAndDescriptors();
		preparePipelines();
		prepareCompute();
		buildCommandBuffers();
		prepared = true;
	}

	virtual void render()
	{
		if (!prepared)
			return;
		draw();

		updateComputeUBO();
	}

	virtual void viewChanged()
	{
		updateGraphicsUBO();
	}

	virtual void keyPressed(uint32_t keyCode)
	{
		switch (keyCode)
		{
		case KEY_W:
		case GAMEPAD_BUTTON_A:
			simulateWind = !simulateWind;
			break;
		}
	}

	virtual void getOverlayText(VulkanTextOverlay *textOverlay)
	{
#if defined(__ANDROID__)
		textOverlay->addText("\"Button A\" to toggle wind simulation", 5.0f, 85.0f, VulkanTextOverlay::alignLeft);
#else
		textOverlay->addText("\"w\" to toggle wind simulation", 5.0f, 85.0f, VulkanTextOverlay::alignLeft);
#endif
	}
};

VULKAN_EXAMPLE_MAIN()