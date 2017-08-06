/*
* Vulkan Example - Dynamic uniform buffers
*
* Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*
* Summary:
* Demonstrates the use of dynamic uniform buffers.
*
* Instead of using one uniform buffer per-object, this example allocates one big uniform buffer
* with respect to the alignment reported by the device via minUniformBufferOffsetAlignment that
* contains all matrices for the objects in the scene.
*
* The used descriptor type vk::DescriptorType::eUniformBufferDynamic then allows to set a dynamic
* offset used to pass data from the single uniform buffer to the connected shader binding point.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <vector>
#include <array>
#include <random>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <vulkan/vulkan.hpp>
#include "vulkanexamplebase.h"
#include "VulkanDevice.hpp"
#include "VulkanBuffer.hpp"

#define VERTEX_BUFFER_BIND_ID 0
#define ENABLE_VALIDATION false
#define OBJECT_INSTANCES 125

// Vertex layout for this example
struct Vertex {
	float pos[3];
	float color[3];
};

// Wrapper functions for aligned memory allocation
// There is currently no standard for this in C++ that works across all platforms and vendors, so we abstract this
void* alignedAlloc(size_t size, size_t alignment)
{
	void *data = nullptr;
#if defined(_MSC_VER) || defined(__MINGW32__)
	data = _aligned_malloc(size, alignment);
#else 
	int res = posix_memalign(&data, alignment, size);
	if (res != 0)
		data = nullptr;
#endif
	return data;
}

void alignedFree(void* data)
{
#if	defined(_MSC_VER) || defined(__MINGW32__)
	_aligned_free(data);
#else 
	free(data);
#endif
}

class VulkanExample : public VulkanExampleBase
{
public:
	struct {
		vk::PipelineVertexInputStateCreateInfo inputState;
		std::vector<vk::VertexInputBindingDescription> bindingDescriptions;
		std::vector<vk::VertexInputAttributeDescription> attributeDescriptions;
	} vertices;

	vks::Buffer vertexBuffer;
	vks::Buffer indexBuffer;
	uint32_t indexCount;

	struct {
		vks::Buffer view;
		vks::Buffer dynamic;
	} uniformBuffers;

	struct {
		glm::mat4 projection;
		glm::mat4 view;
	} uboVS;

	// Store random per-object rotations
	glm::vec3 rotations[OBJECT_INSTANCES];
	glm::vec3 rotationSpeeds[OBJECT_INSTANCES];

	// One big uniform buffer that contains all matrices
	// Note that we need to manually allocate the data to cope for GPU-specific uniform buffer offset alignments
	struct UboDataDynamic {
		glm::mat4 *model = nullptr;
	} uboDataDynamic;

	vk::Pipeline pipeline;
	vk::PipelineLayout pipelineLayout;
	vk::DescriptorSet descriptorSet;
	vk::DescriptorSetLayout descriptorSetLayout;

	float animationTimer = 0.0f;

	size_t dynamicAlignment;

	VulkanExample() : VulkanExampleBase(ENABLE_VALIDATION)
	{
		title = "Vulkan Example - Dynamic uniform buffers";
		enableTextOverlay = true;
		camera.type = Camera::CameraType::lookat;
		camera.setPosition(glm::vec3(0.0f, 0.0f, -30.0f));
		camera.setRotation(glm::vec3(0.0f));
		camera.setPerspective(60.0f, (float)width / (float)height, 0.1f, 256.0f);
	}

	~VulkanExample()
	{
		if (uboDataDynamic.model) {
			alignedFree(uboDataDynamic.model);
		}

		// Clean up used Vulkan resources 
		// Note : Inherited destructor cleans up resources stored in base class
		device.destroyPipeline(pipeline);

		device.destroyPipelineLayout(pipelineLayout);
		device.destroyDescriptorSetLayout(descriptorSetLayout);

		vertexBuffer.destroy();
		indexBuffer.destroy();

		uniformBuffers.view.destroy();
		uniformBuffers.dynamic.destroy();
	}

	void buildCommandBuffers()
	{
		vk::CommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();

		vk::ClearValue clearValues[2];
		clearValues[0].color = defaultClearColor;
		clearValues[1].depthStencil = vk::ClearDepthStencilValue{ 1.0f, 0 };

		vk::RenderPassBeginInfo renderPassBeginInfo = vks::initializers::renderPassBeginInfo();
		renderPassBeginInfo.renderPass = renderPass;
		renderPassBeginInfo.renderArea.offset.x = 0;
		renderPassBeginInfo.renderArea.offset.y = 0;
		renderPassBeginInfo.renderArea.extent.width = width;
		renderPassBeginInfo.renderArea.extent.height = height;
		renderPassBeginInfo.clearValueCount = 2;
		renderPassBeginInfo.pClearValues = clearValues;

		for (uint32_t i = 0; i < drawCmdBuffers.size(); ++i)
		{
			renderPassBeginInfo.framebuffer = frameBuffers[i];

			drawCmdBuffers[i].begin(cmdBufInfo);

			drawCmdBuffers[i].beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);

			vk::Viewport viewport = vks::initializers::viewport((float)width, (float)height, 0.0f, 1.0f);
			drawCmdBuffers[i].setViewport(0, viewport);

			vk::Rect2D scissor = vks::initializers::rect2D(width, height, 0, 0);
			drawCmdBuffers[i].setScissor(0, scissor);

			drawCmdBuffers[i].bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);

			std::vector<vk::DeviceSize> offsets = { 0 };
			drawCmdBuffers[i].bindVertexBuffers(VERTEX_BUFFER_BIND_ID, vertexBuffer.buffer, offsets);
			drawCmdBuffers[i].bindIndexBuffer(indexBuffer.buffer, 0, vk::IndexType::eUint32);

			// Render multiple objects using different model matrices by dynamically offsetting into one uniform buffer
			for (uint32_t j = 0; j < OBJECT_INSTANCES; j++)
			{
				// One dynamic offset per dynamic descriptor to offset into the ubo containing all model matrices
				uint32_t dynamicOffset = j * static_cast<uint32_t>(dynamicAlignment);
				// Bind the descriptor set for rendering a mesh using the dynamic offset
				drawCmdBuffers[i].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSet, dynamicOffset);

				drawCmdBuffers[i].drawIndexed(indexCount, 1, 0, 0, 0);
			}

			drawCmdBuffers[i].endRenderPass();

			drawCmdBuffers[i].end();
		}
	}

	void draw()
	{
		VulkanExampleBase::prepareFrame();

		// Command buffer to be sumitted to the queue
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &drawCmdBuffers[currentBuffer];

		// Submit to queue
		queue.submit(submitInfo, vk::Fence(nullptr));

		VulkanExampleBase::submitFrame();
	}

	void generateCube()
	{
		// Setup vertices indices for a colored cube
		std::vector<Vertex> vertices = {
			{ { -1.0f, -1.0f,  1.0f },{ 1.0f, 0.0f, 0.0f } },
			{ {  1.0f, -1.0f,  1.0f },{ 0.0f, 1.0f, 0.0f } },
			{ {  1.0f,  1.0f,  1.0f },{ 0.0f, 0.0f, 1.0f } },
			{ { -1.0f,  1.0f,  1.0f },{ 0.0f, 0.0f, 0.0f } },
			{ { -1.0f, -1.0f, -1.0f },{ 1.0f, 0.0f, 0.0f } },
			{ {  1.0f, -1.0f, -1.0f },{ 0.0f, 1.0f, 0.0f } },
			{ {  1.0f,  1.0f, -1.0f },{ 0.0f, 0.0f, 1.0f } },
			{ { -1.0f,  1.0f, -1.0f },{ 0.0f, 0.0f, 0.0f } },
		};

		std::vector<uint32_t> indices = { 
			0,1,2, 2,3,0, 1,5,6, 6,2,1, 7,6,5, 5,4,7, 4,0,3, 3,7,4, 4,5,1, 1,0,4, 3,2,6, 6,7,3, 
		};

		indexCount = static_cast<uint32_t>(indices.size());

		// Create buffers
		// For the sake of simplicity we won't stage the vertex data to the gpu memory
		// Vertex buffer
		vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eVertexBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			&vertexBuffer,
			vertices.size() * sizeof(Vertex),
			vertices.data());
		// Index buffer
		vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eIndexBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			&indexBuffer,
			indices.size() * sizeof(uint32_t),
			indices.data());
	}

	void setupVertexDescriptions()
	{
		// Binding description
		vertices.bindingDescriptions = {
			vks::initializers::vertexInputBindingDescription(VERTEX_BUFFER_BIND_ID, sizeof(Vertex), vk::VertexInputRate::eVertex),
		};

		// Attribute descriptions
		vertices.attributeDescriptions = {
			vks::initializers::vertexInputAttributeDescription(VERTEX_BUFFER_BIND_ID, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, pos)),	// Location 0 : Position
			vks::initializers::vertexInputAttributeDescription(VERTEX_BUFFER_BIND_ID, 1, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, color)),	// Location 1 : Color
		};

		vertices.inputState = vks::initializers::pipelineVertexInputStateCreateInfo();
		vertices.inputState.vertexBindingDescriptionCount = static_cast<uint32_t>(vertices.bindingDescriptions.size());
		vertices.inputState.pVertexBindingDescriptions = vertices.bindingDescriptions.data();
		vertices.inputState.vertexAttributeDescriptionCount = static_cast<uint32_t>(vertices.attributeDescriptions.size());
		vertices.inputState.pVertexAttributeDescriptions = vertices.attributeDescriptions.data();
	}

	void setupDescriptorPool()
	{
		// Example uses one ubo and one image sampler
		std::vector<vk::DescriptorPoolSize> poolSizes =
		{
			vks::initializers::descriptorPoolSize(vk::DescriptorType::eUniformBuffer, 1),
			vks::initializers::descriptorPoolSize(vk::DescriptorType::eUniformBufferDynamic, 1),
			vks::initializers::descriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, 1)
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
			vks::initializers::descriptorSetLayoutBinding(vk::DescriptorType::eUniformBuffer, vk::ShaderStageFlagBits::eVertex, 0),
			vks::initializers::descriptorSetLayoutBinding(vk::DescriptorType::eUniformBufferDynamic, vk::ShaderStageFlagBits::eVertex, 1),
			vks::initializers::descriptorSetLayoutBinding(vk::DescriptorType::eCombinedImageSampler, vk::ShaderStageFlagBits::eFragment, 2)
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

		std::vector<vk::WriteDescriptorSet> writeDescriptorSets = {
			// Binding 0 : Projection/View matrix uniform buffer			
			vks::initializers::writeDescriptorSet(descriptorSet, vk::DescriptorType::eUniformBuffer, 0, &uniformBuffers.view.descriptor),
			// Binding 1 : Instance matrix as dynamic uniform buffer
			vks::initializers::writeDescriptorSet(descriptorSet, vk::DescriptorType::eUniformBufferDynamic, 1, &uniformBuffers.dynamic.descriptor),
		};

		device.updateDescriptorSets(writeDescriptorSets, nullptr);
	}

	void preparePipelines()
	{
		vk::PipelineInputAssemblyStateCreateInfo inputAssemblyState =
			vks::initializers::pipelineInputAssemblyStateCreateInfo(
				vk::PrimitiveTopology::eTriangleList,
				vk::PipelineInputAssemblyStateCreateFlags(),
				VK_FALSE);

		vk::PipelineRasterizationStateCreateInfo rasterizationState =
			vks::initializers::pipelineRasterizationStateCreateInfo(
				vk::PolygonMode::eFill,
				vk::CullModeFlagBits::eNone,
				vk::FrontFace::eCounterClockwise);

		vk::PipelineColorBlendAttachmentState blendAttachmentState =
			vks::initializers::pipelineColorBlendAttachmentState(
				vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
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
			vks::initializers::pipelineViewportStateCreateInfo(1, 1);

		vk::PipelineMultisampleStateCreateInfo multisampleState =
			vks::initializers::pipelineMultisampleStateCreateInfo(vk::SampleCountFlagBits::e1);

		std::vector<vk::DynamicState> dynamicStateEnables = {
			vk::DynamicState::eViewport,
			vk::DynamicState::eScissor
		};
		vk::PipelineDynamicStateCreateInfo dynamicState =
			vks::initializers::pipelineDynamicStateCreateInfo(
				dynamicStateEnables);

		// Load shaders
		std::array<vk::PipelineShaderStageCreateInfo, 2> shaderStages;

		shaderStages[0] = loadShader(getAssetPath() + "shaders/dynamicuniformbuffer/base.vert.spv", vk::ShaderStageFlagBits::eVertex);
		shaderStages[1] = loadShader(getAssetPath() + "shaders/dynamicuniformbuffer/base.frag.spv", vk::ShaderStageFlagBits::eFragment);

		vk::GraphicsPipelineCreateInfo pipelineCreateInfo =
			vks::initializers::pipelineCreateInfo(
				pipelineLayout,
				renderPass);

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

		pipeline = device.createGraphicsPipelines(pipelineCache, pipelineCreateInfo)[0];
	}

	// Prepare and initialize uniform buffer containing shader uniforms
	void prepareUniformBuffers()
	{
		// Allocate data for the dynamic uniform buffer object
		// We allocate this manually as the alignment of the offset differs between GPUs

		// Calculate required alignment depending on device limits
		size_t uboAlignment = vulkanDevice->properties.limits.minUniformBufferOffsetAlignment;
		dynamicAlignment = (sizeof(glm::mat4) / uboAlignment) * uboAlignment + ((sizeof(glm::mat4) % uboAlignment) > 0 ? uboAlignment : 0);

		size_t bufferSize = OBJECT_INSTANCES * dynamicAlignment;

		uboDataDynamic.model = (glm::mat4*)alignedAlloc(bufferSize, dynamicAlignment);
		assert(uboDataDynamic.model);

		std::cout << "minUniformBufferOffsetAlignment = " << uboAlignment << std::endl;
		std::cout << "dynamicAlignment = " << dynamicAlignment << std::endl;

		// Vertex shader uniform buffer block

		// Static shared uniform buffer object with projection and view matrix
		vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eUniformBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			&uniformBuffers.view,
			sizeof(uboVS));

		// Uniform buffer object with per-object matrices
		vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eUniformBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible,
			&uniformBuffers.dynamic,
			bufferSize);

		// Map persistent
		uniformBuffers.view.map();
		uniformBuffers.dynamic.map();

		// Prepare per-object matrices with offsets and random rotations
		std::mt19937 rndGen(static_cast<uint32_t>(time(0)));
		std::normal_distribution<float> rndDist(-1.0f, 1.0f);
		for (uint32_t i = 0; i < OBJECT_INSTANCES; i++)
		{
			rotations[i] = glm::vec3(rndDist(rndGen), rndDist(rndGen), rndDist(rndGen)) * 2.0f * (float)M_PI;
			rotationSpeeds[i] = glm::vec3(rndDist(rndGen), rndDist(rndGen), rndDist(rndGen));
		}

		updateUniformBuffers();
		updateDynamicUniformBuffer(true);
	}

	void updateUniformBuffers()
	{
		// Fixed ubo with projection and view matrices
		uboVS.projection = camera.matrices.perspective;
		uboVS.view = camera.matrices.view;

		memcpy(uniformBuffers.view.mapped, &uboVS, sizeof(uboVS));
	}

	void updateDynamicUniformBuffer(bool force = false)
	{
		// Update at max. 60 fps
		animationTimer += frameTimer;
		if ((animationTimer <= 1.0f / 60.0f) && (!force)) {
			return;
		}

		// Dynamic ubo with per-object model matrices indexed by offsets in the command buffer
		uint32_t dim = static_cast<uint32_t>(pow(OBJECT_INSTANCES, (1.0f / 3.0f)));
		glm::vec3 offset(5.0f);

		for (uint32_t x = 0; x < dim; x++)
		{
			for (uint32_t y = 0; y < dim; y++)
			{
				for (uint32_t z = 0; z < dim; z++)
				{
					uint32_t index = x * dim * dim + y * dim + z;

					// Aligned offset
					glm::mat4* modelMat = (glm::mat4*)(((uint64_t)uboDataDynamic.model + (index * dynamicAlignment)));

					// Update rotations
					rotations[index] += animationTimer * rotationSpeeds[index];

					// Update matrices
					glm::vec3 pos = glm::vec3(-((dim * offset.x) / 2.0f) + offset.x / 2.0f + x * offset.x, -((dim * offset.y) / 2.0f) + offset.y / 2.0f + y * offset.y, -((dim * offset.z) / 2.0f) + offset.z / 2.0f + z * offset.z);
					*modelMat = glm::translate(glm::mat4(), pos);
					*modelMat = glm::rotate(*modelMat, rotations[index].x, glm::vec3(1.0f, 1.0f, 0.0f));
					*modelMat = glm::rotate(*modelMat, rotations[index].y, glm::vec3(0.0f, 1.0f, 0.0f));
					*modelMat = glm::rotate(*modelMat, rotations[index].z, glm::vec3(0.0f, 0.0f, 1.0f));
				}
			}
		}

		animationTimer = 0.0f;

		memcpy(uniformBuffers.dynamic.mapped, uboDataDynamic.model, uniformBuffers.dynamic.size);
		// Flush to make changes visible to the host 
		vk::MappedMemoryRange memoryRange = vks::initializers::mappedMemoryRange();
		memoryRange.memory = uniformBuffers.dynamic.memory;
		memoryRange.size = uniformBuffers.dynamic.size;
		device.flushMappedMemoryRanges(memoryRange);
	}

	void prepare()
	{
		VulkanExampleBase::prepare();
		generateCube();
		setupVertexDescriptions();
		prepareUniformBuffers();
		setupDescriptorSetLayout();
		preparePipelines();
		setupDescriptorPool();
		setupDescriptorSet();
		buildCommandBuffers();
		prepared = true;
	}

	virtual void render()
	{
		if (!prepared)
			return;
		draw();
		if (!paused)
			updateDynamicUniformBuffer();
	}

	virtual void viewChanged()
	{
		updateUniformBuffers();
	}
};

VULKAN_EXAMPLE_MAIN()
