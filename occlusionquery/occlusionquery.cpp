/*
* Vulkan Example - Using occlusion query for visbility testing
*
* Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <vector>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <vulkan/vulkan.hpp>
#include "vulkanexamplebase.h"
#include "VulkanBuffer.hpp"
#include "VulkanModel.hpp"

#define VERTEX_BUFFER_BIND_ID 0
#define ENABLE_VALIDATION false

class VulkanExample : public VulkanExampleBase
{
public:
	// Vertex layout for the models
	vks::VertexLayout vertexLayout = vks::VertexLayout({
		vks::VERTEX_COMPONENT_POSITION,
		vks::VERTEX_COMPONENT_NORMAL,
		vks::VERTEX_COMPONENT_COLOR,
	});

	struct {
		vks::Model teapot;
		vks::Model plane;
		vks::Model sphere;
	} models;

	struct {
		vks::Buffer occluder;
		vks::Buffer teapot;
		vks::Buffer sphere;
	} uniformBuffers;

	struct UBOVS {
		glm::mat4 projection;
		glm::mat4 model;
		glm::vec4 lightPos = glm::vec4(10.0f, 10.0f, 10.0f, 1.0f);
		float visible;
	} uboVS;

	struct {
		vk::PipelineVertexInputStateCreateInfo inputState;
		std::vector<vk::VertexInputBindingDescription> bindingDescriptions;
		std::vector<vk::VertexInputAttributeDescription> attributeDescriptions;
	} vertices;

	struct {
		vk::Pipeline solid;
		vk::Pipeline occluder;
		// Pipeline with basic shaders used for occlusion pass
		vk::Pipeline simple;
	} pipelines;

	struct {
		vk::DescriptorSet teapot;
		vk::DescriptorSet sphere;
	} descriptorSets;

	vk::PipelineLayout pipelineLayout;
	vk::DescriptorSet descriptorSet;
	vk::DescriptorSetLayout descriptorSetLayout;

	// Stores occlusion query results
	struct {
		vk::Buffer buffer;
		vk::DeviceMemory memory;
	} queryResult;

	// Pool that stores all occlusion queries
	vk::QueryPool queryPool;

	// Passed query samples
	uint64_t passedSamples[2] = { 1,1 };

	VulkanExample() : VulkanExampleBase(ENABLE_VALIDATION)
	{
		zoom = -35.0f;
		zoomSpeed = 2.5f;
		rotationSpeed = 0.5f;
		rotation = { 0.0, -123.75, 0.0 };
		enableTextOverlay = true;
		title = "Vulkan Example - Occlusion queries";
	}

	~VulkanExample()
	{
		// Clean up used Vulkan resources 
		// Note : Inherited destructor cleans up resources stored in base class
		device.destroyPipeline(pipelines.solid);
		device.destroyPipeline(pipelines.occluder);
		device.destroyPipeline(pipelines.simple);

		device.destroyPipelineLayout(pipelineLayout);
		device.destroyDescriptorSetLayout(descriptorSetLayout);

		vkDestroyQueryPool(device, queryPool, nullptr);

		device.destroyBuffer(queryResult.buffer);
		device.freeMemory(queryResult.memory);

		uniformBuffers.occluder.destroy();
		uniformBuffers.sphere.destroy();
		uniformBuffers.teapot.destroy();

		models.sphere.destroy();
		models.plane.destroy();
		models.teapot.destroy();
	}

	// Create a buffer for storing the query result
	// Setup a query pool
	void setupQueryResultBuffer()
	{
		uint32_t bufSize = 2 * sizeof(uint64_t);

		vk::MemoryRequirements memReqs;
		vk::MemoryAllocateInfo memAlloc = vks::initializers::memoryAllocateInfo();
		vk::BufferCreateInfo bufferCreateInfo = 
			vks::initializers::bufferCreateInfo(
				vk::BufferUsageFlagBits::eUniformBuffer | vk::BufferUsageFlagBits::eTransferDst, 
				bufSize);

		// Results are saved in a host visible buffer for easy access by the application
		queryResult.buffer = device.createBuffer(bufferCreateInfo);
		memReqs = device.getBufferMemoryRequirements(queryResult.buffer);
		memAlloc.allocationSize = memReqs.size;
		memAlloc.memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
		&queryResult.memory = device.allocateMemory(memAlloc);
		device.bindBufferMemory(queryResult.buffer, queryResult.memory, 0);

		// Create query pool
		vk::QueryPoolCreateInfo queryPoolInfo = {};

		// Query pool will be created for occlusion queries
		queryPoolInfo.queryType = vk::QueryType::eOcclusion;
		queryPoolInfo.queryCount = 2;

		VK_CHECK_RESULT(vkCreateQueryPool(device, &queryPoolInfo, NULL, &queryPool));
	}

	// Retrieves the results of the occlusion queries submitted to the command buffer
	void getQueryResults()
	{
		// We use vkGetQueryResults to copy the results into a host visible buffer
		vkGetQueryPoolResults(
			device, 
			queryPool,
			0,
			2,
			sizeof(passedSamples),
			passedSamples,
			sizeof(uint64_t),
			// Store results a 64 bit values and wait until the results have been finished
			// If you don't want to wait, you can use vk::QueryResultFlagBits::eWithAvailability
			// which also returns the state of the result (ready) in the result
			vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait);
	}

	void buildCommandBuffers()
	{
		vk::CommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();

		vk::ClearValue clearValues[2];
		clearValues[0].color = defaultClearColor;
		clearValues[1].depthStencil = { 1.0f, 0 };

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
			// Set target frame buffer
			renderPassBeginInfo.framebuffer = frameBuffers[i];

			drawCmdBuffers[i].begin(cmdBufInfo);

			// Reset query pool
			// Must be done outside of render pass
			vkCmdResetQueryPool(drawCmdBuffers[i], queryPool, 0, 2);

			drawCmdBuffers[i].beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);

			vk::Viewport viewport = vks::initializers::viewport(
				(float)width,
				(float)height,
				0.0f,
				1.0f);
			drawCmdBuffers[i].setViewport(0, viewport);

			vk::Rect2D scissor = vks::initializers::rect2D(
				width,
				height,
				0,
				0);
			vkCmdSetScissor(drawCmdBuffers[i], 0, 1, &scissor);

			vk::DeviceSize offsets[1] = { 0 };

			//glm::mat4 modelMatrix = glm::mat4();

			// Occlusion pass
			drawCmdBuffers[i].bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.simple);

			// Occluder first
			drawCmdBuffers[i].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSet, nullptr);
			drawCmdBuffers[i].bindVertexBuffers(VERTEX_BUFFER_BIND_ID, models.plane.vertices.buffer, offsets);
			drawCmdBuffers[i].bindIndexBuffer(models.plane.indices.buffer, 0, vk::IndexType::eUint32);
			drawCmdBuffers[i].drawIndexed(models.plane.indexCount, 1, 0, 0, 0);

			// Teapot
			vkCmdBeginQuery(drawCmdBuffers[i], queryPool, 0, VK_FLAGS_NONE);

			drawCmdBuffers[i].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSets.teapot, nullptr);
			drawCmdBuffers[i].bindVertexBuffers(VERTEX_BUFFER_BIND_ID, models.teapot.vertices.buffer, offsets);
			drawCmdBuffers[i].bindIndexBuffer(models.teapot.indices.buffer, 0, vk::IndexType::eUint32);
			drawCmdBuffers[i].drawIndexed(models.teapot.indexCount, 1, 0, 0, 0);

			vkCmdEndQuery(drawCmdBuffers[i], queryPool, 0);

			// Sphere
			vkCmdBeginQuery(drawCmdBuffers[i], queryPool, 1, VK_FLAGS_NONE);

			drawCmdBuffers[i].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSets.sphere, nullptr);
			drawCmdBuffers[i].bindVertexBuffers(VERTEX_BUFFER_BIND_ID, models.sphere.vertices.buffer, offsets);
			drawCmdBuffers[i].bindIndexBuffer(models.sphere.indices.buffer, 0, vk::IndexType::eUint32);
			drawCmdBuffers[i].drawIndexed(models.sphere.indexCount, 1, 0, 0, 0);

			vkCmdEndQuery(drawCmdBuffers[i], queryPool, 1);

			// Visible pass
			// Clear color and depth attachments
			vk::ClearAttachment clearAttachments[2] = {};

			clearAttachments[0].aspectMask = vk::ImageAspectFlagBits::eColor;
			clearAttachments[0].clearValue.color = defaultClearColor;
			clearAttachments[0].colorAttachment = 0;

			clearAttachments[1].aspectMask = vk::ImageAspectFlagBits::eDepth;
			clearAttachments[1].clearValue.depthStencil = { 1.0f, 0 };

			vk::ClearRect clearRect = {};
			clearRect.layerCount = 1;
			clearRect.rect.offset = { 0, 0 };
			clearRect.rect.extent = { width, height };

			vkCmdClearAttachments(
				drawCmdBuffers[i],
				2,
				clearAttachments,
				1,
				&clearRect);

			drawCmdBuffers[i].bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.solid);

			// Teapot
			drawCmdBuffers[i].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSets.teapot, nullptr);
			drawCmdBuffers[i].bindVertexBuffers(VERTEX_BUFFER_BIND_ID, models.teapot.vertices.buffer, offsets);
			drawCmdBuffers[i].bindIndexBuffer(models.teapot.indices.buffer, 0, vk::IndexType::eUint32);
			drawCmdBuffers[i].drawIndexed(models.teapot.indexCount, 1, 0, 0, 0);

			// Sphere
			drawCmdBuffers[i].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSets.sphere, nullptr);
			drawCmdBuffers[i].bindVertexBuffers(VERTEX_BUFFER_BIND_ID, models.sphere.vertices.buffer, offsets);
			drawCmdBuffers[i].bindIndexBuffer(models.sphere.indices.buffer, 0, vk::IndexType::eUint32);
			drawCmdBuffers[i].drawIndexed(models.sphere.indexCount, 1, 0, 0, 0);

			// Occluder
			drawCmdBuffers[i].bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.occluder);
			drawCmdBuffers[i].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSet, nullptr);
			drawCmdBuffers[i].bindVertexBuffers(VERTEX_BUFFER_BIND_ID, models.plane.vertices.buffer, offsets);
			drawCmdBuffers[i].bindIndexBuffer(models.plane.indices.buffer, 0, vk::IndexType::eUint32);
			drawCmdBuffers[i].drawIndexed(models.plane.indexCount, 1, 0, 0, 0);

			drawCmdBuffers[i].endRenderPass();

			drawCmdBuffers[i].end();
		}
	}

	void draw()
	{
		VulkanExampleBase::prepareFrame();

		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &drawCmdBuffers[currentBuffer];
		queue.submit(submitInfo, vk::Fence(nullptr));

		// Read query results for displaying in next frame
		getQueryResults();

		VulkanExampleBase::submitFrame();
	}

	void loadAssets()
	{
		models.plane.loadFromFile(getAssetPath() + "models/plane_z.3ds", vertexLayout, 0.4f, vulkanDevice, queue);
		models.teapot.loadFromFile(getAssetPath() + "models/teapot.3ds", vertexLayout, 0.3f, vulkanDevice, queue);
		models.sphere.loadFromFile(getAssetPath() + "models/sphere.3ds", vertexLayout, 0.3f, vulkanDevice, queue);
	}

	void setupVertexDescriptions()
	{
		// Binding description
		vertices.bindingDescriptions.resize(1);
		vertices.bindingDescriptions[0] =
			vks::initializers::vertexInputBindingDescription(
				VERTEX_BUFFER_BIND_ID,
				vertexLayout.stride(),
				vk::VertexInputRate::eVertex);

		// Attribute descriptions
		// Describes memory layout and shader positions
		vertices.attributeDescriptions.resize(3);
		// Location 0 : Position
		vertices.attributeDescriptions[0] =
			vks::initializers::vertexInputAttributeDescription(
				VERTEX_BUFFER_BIND_ID,
				0,
				vk::Format::eR32G32B32Sfloat,
				0);
		// Location 1 : Normal
		vertices.attributeDescriptions[1] =
			vks::initializers::vertexInputAttributeDescription(
				VERTEX_BUFFER_BIND_ID,
				1,
				vk::Format::eR32G32B32Sfloat,
				sizeof(float) * 3);
		// Location 3 : Color
		vertices.attributeDescriptions[2] =
			vks::initializers::vertexInputAttributeDescription(
				VERTEX_BUFFER_BIND_ID,
				2,
				vk::Format::eR32G32B32Sfloat,
				sizeof(float) * 6);

		vertices.inputState = vks::initializers::pipelineVertexInputStateCreateInfo();
		vertices.inputState.vertexBindingDescriptionCount = vertices.bindingDescriptions.size();
		vertices.inputState.pVertexBindingDescriptions = vertices.bindingDescriptions.data();
		vertices.inputState.vertexAttributeDescriptionCount = vertices.attributeDescriptions.size();
		vertices.inputState.pVertexAttributeDescriptions = vertices.attributeDescriptions.data();
	}

	void setupDescriptorPool()
	{
		std::vector<vk::DescriptorPoolSize> poolSizes =
		{
			// One uniform buffer block for each mesh
			vks::initializers::descriptorPoolSize(vk::DescriptorType::eUniformBuffer, 3)
		};

		vk::DescriptorPoolCreateInfo descriptorPoolInfo =
			vks::initializers::descriptorPoolCreateInfo(
				poolSizes.size(),
				poolSizes.data(),
				3);

		descriptorPool = device.createDescriptorPool(descriptorPoolInfo);
	}

	void setupDescriptorSetLayout()
	{
		std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings =
		{
			// Binding 0 : Vertex shader uniform buffer
			vks::initializers::descriptorSetLayoutBinding(
				vk::DescriptorType::eUniformBuffer,
				vk::ShaderStageFlagBits::eVertex,
				0)
		};

		vk::DescriptorSetLayoutCreateInfo descriptorLayout =
			vks::initializers::descriptorSetLayoutCreateInfo(
				setLayoutBindings.data(),
				setLayoutBindings.size());

		descriptorSetLayout = device.createDescriptorSetLayout(descriptorLayout);

		vk::PipelineLayoutCreateInfo pPipelineLayoutCreateInfo =
			vks::initializers::pipelineLayoutCreateInfo(
				&descriptorSetLayout,
				1);

		pipelineLayout = device.createPipelineLayout(pPipelineLayoutCreateInfo);
	}

	void setupDescriptorSets()
	{
		vk::DescriptorSetAllocateInfo allocInfo =
			vks::initializers::descriptorSetAllocateInfo(
				descriptorPool,
				&descriptorSetLayout,
				1);

		// Occluder (plane)
		descriptorSet = device.allocateDescriptorSets(allocInfo)[0];

		std::vector<vk::WriteDescriptorSet> writeDescriptorSets =
		{
			// Binding 0 : Vertex shader uniform buffer
			vks::initializers::writeDescriptorSet(
				descriptorSet,
				vk::DescriptorType::eUniformBuffer,
				0,
				&uniformBuffers.occluder.descriptor)
		};

		vkUpdateDescriptorSets(device, writeDescriptorSets.size(), writeDescriptorSets.data(), 0, NULL);

		// Teapot
		descriptorSets.teapot = device.allocateDescriptorSets(allocInfo)[0];
		writeDescriptorSets[0].dstSet = descriptorSets.teapot;
		writeDescriptorSets[0].pBufferInfo = &uniformBuffers.teapot.descriptor;
		vkUpdateDescriptorSets(device, writeDescriptorSets.size(), writeDescriptorSets.data(), 0, NULL);

		// Sphere
		descriptorSets.sphere = device.allocateDescriptorSets(allocInfo)[0];
		writeDescriptorSets[0].dstSet = descriptorSets.sphere;
		writeDescriptorSets[0].pBufferInfo = &uniformBuffers.sphere.descriptor;
		vkUpdateDescriptorSets(device, writeDescriptorSets.size(), writeDescriptorSets.data(), 0, NULL);
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
				dynamicStateEnables.size(),
				0);

		// Solid rendering pipeline
		// Load shaders
		std::array<vk::PipelineShaderStageCreateInfo, 2> shaderStages;

		shaderStages[0] = loadShader(getAssetPath() + "shaders/occlusionquery/mesh.vert.spv", vk::ShaderStageFlagBits::eVertex);
		shaderStages[1] = loadShader(getAssetPath() + "shaders/occlusionquery/mesh.frag.spv", vk::ShaderStageFlagBits::eFragment);

		vk::GraphicsPipelineCreateInfo pipelineCreateInfo =
			vks::initializers::pipelineCreateInfo(
				pipelineLayout,
				renderPass,
				0);

		pipelineCreateInfo.pVertexInputState = &vertices.inputState;
		pipelineCreateInfo.pInputAssemblyState = &inputAssemblyState;
		pipelineCreateInfo.pRasterizationState = &rasterizationState;
		pipelineCreateInfo.pColorBlendState = &colorBlendState;
		pipelineCreateInfo.pMultisampleState = &multisampleState;
		pipelineCreateInfo.pViewportState = &viewportState;
		pipelineCreateInfo.pDepthStencilState = &depthStencilState;
		pipelineCreateInfo.pDynamicState = &dynamicState;
		pipelineCreateInfo.stageCount = shaderStages.size();
		pipelineCreateInfo.pStages = shaderStages.data();

		pipelines.solid = device.createGraphicsPipelines(pipelineCache, pipelineCreateInfo)[0];

		// Basic pipeline for coloring occluded objects
		shaderStages[0] = loadShader(getAssetPath() + "shaders/occlusionquery/simple.vert.spv", vk::ShaderStageFlagBits::eVertex);
		shaderStages[1] = loadShader(getAssetPath() + "shaders/occlusionquery/simple.frag.spv", vk::ShaderStageFlagBits::eFragment);
		rasterizationState.cullMode = vk::CullModeFlagBits::eNone;

		pipelines.simple = device.createGraphicsPipelines(pipelineCache, pipelineCreateInfo)[0];

		// Visual pipeline for the occluder
		shaderStages[0] = loadShader(getAssetPath() + "shaders/occlusionquery/occluder.vert.spv", vk::ShaderStageFlagBits::eVertex);
		shaderStages[1] = loadShader(getAssetPath() + "shaders/occlusionquery/occluder.frag.spv", vk::ShaderStageFlagBits::eFragment);

		// Enable blending
		blendAttachmentState.blendEnable = VK_TRUE;
		blendAttachmentState.colorBlendOp = vk::BlendOp::eAdd;
		blendAttachmentState.srcColorBlendFactor = vk::BlendFactor::eSrcColor;
		blendAttachmentState.dstColorBlendFactor = vk::BlendFactor::eOne_MINUS_SRC_COLOR;

		pipelines.occluder = device.createGraphicsPipelines(pipelineCache, pipelineCreateInfo)[0];
	}

	// Prepare and initialize uniform buffer containing shader uniforms
	void prepareUniformBuffers()
	{
		// Vertex shader uniform buffer block
		VK_CHECK_RESULT(vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eUniformBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			&uniformBuffers.occluder,
			sizeof(uboVS)));

		// Teapot
		VK_CHECK_RESULT(vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eUniformBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			&uniformBuffers.teapot,
			sizeof(uboVS)));

		// Sphere
		VK_CHECK_RESULT(vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eUniformBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			&uniformBuffers.sphere,
			sizeof(uboVS)));

		// Map persistent
		uniformBuffers.occluder.map();
		uniformBuffers.teapot.map();
		uniformBuffers.sphere.map();

		updateUniformBuffers();
	}

	void updateUniformBuffers()
	{
		// Vertex shader
		uboVS.projection = glm::perspective(glm::radians(60.0f), (float)width / (float)height, 0.1f, 256.0f);
		glm::mat4 viewMatrix = glm::translate(glm::mat4(), glm::vec3(0.0f, 0.0f, zoom));

		glm::mat4 rotMatrix = glm::mat4();
		rotMatrix = glm::rotate(rotMatrix, glm::radians(rotation.x), glm::vec3(1.0f, 0.0f, 0.0f));
		rotMatrix = glm::rotate(rotMatrix, glm::radians(rotation.y), glm::vec3(0.0f, 1.0f, 0.0f));
		rotMatrix = glm::rotate(rotMatrix, glm::radians(rotation.z), glm::vec3(0.0f, 0.0f, 1.0f));

		uboVS.model = viewMatrix * rotMatrix;

		// Occluder
		uboVS.visible = 1.0f;
		memcpy(uniformBuffers.occluder.mapped, &uboVS, sizeof(uboVS));

		// Teapot
		// Toggle color depending on visibility
		uboVS.visible = (passedSamples[0] > 0) ? 1.0f : 0.0f;
		uboVS.model = viewMatrix * rotMatrix * glm::translate(glm::mat4(), glm::vec3(0.0f, 0.0f, -10.0f));
		memcpy(uniformBuffers.teapot.mapped, &uboVS, sizeof(uboVS));

		// Sphere
		// Toggle color depending on visibility
		uboVS.visible = (passedSamples[1] > 0) ? 1.0f : 0.0f;
		uboVS.model = viewMatrix * rotMatrix * glm::translate(glm::mat4(), glm::vec3(0.0f, 0.0f, 10.0f));
		memcpy(uniformBuffers.sphere.mapped, &uboVS, sizeof(uboVS));
	}

	void prepare()
	{
		VulkanExampleBase::prepare();
		loadAssets();
		setupQueryResultBuffer();
		setupVertexDescriptions();
		prepareUniformBuffers();
		setupDescriptorSetLayout();
		preparePipelines();
		setupDescriptorPool();
		setupDescriptorSets();
		buildCommandBuffers();
		prepared = true;
	}

	virtual void render()
	{
		if (!prepared)
			return;
		draw();
	}

	virtual void viewChanged()
	{
		vkDeviceWaitIdle(device);
		updateUniformBuffers();
		VulkanExampleBase::updateTextOverlay();
	}

	virtual void getOverlayText(VulkanTextOverlay *textOverlay)
	{
		textOverlay->addText("Occlusion queries:", 5.0f, 85.0f, VulkanTextOverlay::alignLeft);
		textOverlay->addText("Teapot: " + std::to_string(passedSamples[0]) + " samples passed" , 5.0f, 105.0f, VulkanTextOverlay::alignLeft);
		textOverlay->addText("Sphere: " + std::to_string(passedSamples[1]) + " samples passed", 5.0f, 125.0f, VulkanTextOverlay::alignLeft);
	}
};

VULKAN_EXAMPLE_MAIN()