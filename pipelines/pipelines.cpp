/*
* Vulkan Example - Using different pipelines in one single renderpass
*
* Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include "VulkanExampleBase.hpp"
#include "VulkanModel.hpp"

#define VERTEX_BUFFER_BIND_ID 0
#define ENABLE_VALIDATION false

class VulkanExample: public VulkanExampleBase 
{
public:
	// Vertex layout for the models
	vks::VertexLayout vertexLayout = vks::VertexLayout({
		vks::VERTEX_COMPONENT_POSITION,
		vks::VERTEX_COMPONENT_NORMAL,
		vks::VERTEX_COMPONENT_UV,
		vks::VERTEX_COMPONENT_COLOR,
	});

	struct {
		vks::Model cube;
	} models;

	vks::Buffer uniformBuffer;

	// Same uniform buffer layout as shader
	struct UBOVS {
		glm::mat4 projection;
		glm::mat4 modelView;
		glm::vec4 lightPos = glm::vec4(0.0f, 2.0f, 1.0f, 0.0f);
	} uboVS;

	vk::PipelineLayout pipelineLayout;
	vk::DescriptorSet descriptorSet;
	vk::DescriptorSetLayout descriptorSetLayout;

	struct {
		vk::Pipeline phong;
		vk::Pipeline wireframe;
		vk::Pipeline toon;
	} pipelines;

	VulkanExample() : VulkanExampleBase(ENABLE_VALIDATION)
	{
		zoom = -10.5f;
		rotation = glm::vec3(-25.0f, 15.0f, 0.0f);
		enableTextOverlay = true;
		title = "Vulkan Example - Pipeline state objects";
	}

	~VulkanExample()
	{
		// Clean up used Vulkan resources 
		// Note : Inherited destructor cleans up resources stored in base class
		device.destroyPipeline(pipelines.phong);
		if (deviceFeatures.fillModeNonSolid)
		{
			device.destroyPipeline(pipelines.wireframe);
		}
		device.destroyPipeline(pipelines.toon);
		
		device.destroyPipelineLayout(pipelineLayout);
		device.destroyDescriptorSetLayout(descriptorSetLayout);

		models.cube.destroy();
		uniformBuffer.destroy();
	}

	// Enable physical device features required for this example				
	virtual void getEnabledFeatures()
	{
		// Fill mode non solid is required for wireframe display
		if (deviceFeatures.fillModeNonSolid) {
			enabledFeatures.fillModeNonSolid = VK_TRUE;
			// Wide lines must be present for line width > 1.0f
			if (deviceFeatures.wideLines) {
				enabledFeatures.wideLines = VK_TRUE;
			}
		};
	}

	void buildCommandBuffers()
	{		 
		vk::CommandBufferBeginInfo cmdBufInfo;

		vk::ClearValue clearValues[2];
		clearValues[0].color = defaultClearColor;
		clearValues[1].depthStencil = vk::ClearDepthStencilValue{ 1.0f, 0 };

		vk::RenderPassBeginInfo renderPassBeginInfo;
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

			drawCmdBuffers[i].beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);

			vk::Viewport viewport = vks::initializers::viewport((float)width, (float)height, 0.0f, 1.0f);
			drawCmdBuffers[i].setViewport(0, viewport);

			vk::Rect2D scissor = vks::initializers::rect2D(width, height,	0, 0);
			drawCmdBuffers[i].setScissor(0, scissor);

			drawCmdBuffers[i].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSet, nullptr);

			std::vector<vk::DeviceSize> offsets = { 0 };
			drawCmdBuffers[i].bindVertexBuffers(VERTEX_BUFFER_BIND_ID, models.cube.vertices.buffer, offsets);
			drawCmdBuffers[i].bindIndexBuffer(models.cube.indices.buffer, 0, vk::IndexType::eUint32);

			// Left : Solid colored 
			viewport.width = (float)width / 3.0;
			drawCmdBuffers[i].setViewport(0, viewport);
			drawCmdBuffers[i].bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.phong);
			
			drawCmdBuffers[i].drawIndexed(models.cube.indexCount, 1, 0, 0, 0);

			// Center : Toon
			viewport.x = (float)width / 3.0;
			drawCmdBuffers[i].setViewport(0, viewport);
			drawCmdBuffers[i].bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.toon);
			// Line width > 1.0f only if wide lines feature is supported
			if (deviceFeatures.wideLines) {
				vkCmdSetLineWidth(drawCmdBuffers[i], 2.0f);
			}
			drawCmdBuffers[i].drawIndexed(models.cube.indexCount, 1, 0, 0, 0);

			if (deviceFeatures.fillModeNonSolid)
			{
				// Right : Wireframe 
				viewport.x = (float)width / 3.0 + (float)width / 3.0;
				drawCmdBuffers[i].setViewport(0, viewport);
				drawCmdBuffers[i].bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.wireframe);
				drawCmdBuffers[i].drawIndexed(models.cube.indexCount, 1, 0, 0, 0);
			}

			drawCmdBuffers[i].endRenderPass();

			drawCmdBuffers[i].end();
		}
	}

	void loadAssets()
	{
		models.cube.loadFromFile(getAssetPath() + "models/treasure_smooth.dae", vertexLayout, 1.0f, vulkanDevice, queue);
	}

	void setupDescriptorPool()
	{
		std::vector<vk::DescriptorPoolSize> poolSizes =
		{
			vks::initializers::descriptorPoolSize(vk::DescriptorType::eUniformBuffer, 1)
		};

		vk::DescriptorPoolCreateInfo descriptorPoolInfo =
			vks::initializers::descriptorPoolCreateInfo(
				poolSizes.size(),
				poolSizes.data(),
				2);

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
			// Binding 0 : Vertex shader uniform buffer
			vks::initializers::writeDescriptorSet(
				descriptorSet,
				vk::DescriptorType::eUniformBuffer,
				0,
				&uniformBuffer.descriptor)
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
				vk::CullModeFlagBits::eBack,
				vk::FrontFace::eClockwise);

		vk::PipelineColorBlendAttachmentState blendAttachmentState =
			vks::initializers::pipelineColorBlendAttachmentState();

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
			vk::DynamicState::eScissor,
			vk::DynamicState::eLineWidth,
		};
		vk::PipelineDynamicStateCreateInfo dynamicState =
			vks::initializers::pipelineDynamicStateCreateInfo(dynamicStateEnables);

		vk::GraphicsPipelineCreateInfo pipelineCreateInfo =
			vks::initializers::pipelineCreateInfo(pipelineLayout, renderPass);

		std::array<vk::PipelineShaderStageCreateInfo, 2> shaderStages;

		pipelineCreateInfo.pInputAssemblyState = &inputAssemblyState;
		pipelineCreateInfo.pRasterizationState = &rasterizationState;
		pipelineCreateInfo.pColorBlendState = &colorBlendState;
		pipelineCreateInfo.pMultisampleState = &multisampleState;
		pipelineCreateInfo.pViewportState = &viewportState;
		pipelineCreateInfo.pDepthStencilState = &depthStencilState;
		pipelineCreateInfo.pDynamicState = &dynamicState;
		pipelineCreateInfo.stageCount = shaderStages.size();
		pipelineCreateInfo.pStages = shaderStages.data();

		// Shared vertex bindings and attributes used by all pipelines

		// Binding description
		std::vector<vk::VertexInputBindingDescription> vertexInputBindings = {
			vks::initializers::vertexInputBindingDescription(VERTEX_BUFFER_BIND_ID, vertexLayout.stride(), vk::VertexInputRate::eVertex),
		};

		// Attribute descriptions
		std::vector<vk::VertexInputAttributeDescription> vertexInputAttributes = {
			vks::initializers::vertexInputAttributeDescription(VERTEX_BUFFER_BIND_ID, 0, vk::Format::eR32G32B32Sfloat, 0),					// Location 0: Position			
			vks::initializers::vertexInputAttributeDescription(VERTEX_BUFFER_BIND_ID, 1, vk::Format::eR32G32B32Sfloat, sizeof(float) * 3),	// Location 1: Color			
			vks::initializers::vertexInputAttributeDescription(VERTEX_BUFFER_BIND_ID, 2, vk::Format::eR32G32Sfloat, sizeof(float) * 6),		// Location 2 : Texture coordinates			
			vks::initializers::vertexInputAttributeDescription(VERTEX_BUFFER_BIND_ID, 3, vk::Format::eR32G32B32Sfloat, sizeof(float) * 8),	// Location 3 : Normal
		};

		vk::PipelineVertexInputStateCreateInfo vertexInputState = vks::initializers::pipelineVertexInputStateCreateInfo();
		vertexInputState.vertexBindingDescriptionCount = static_cast<uint32_t>(vertexInputBindings.size());
		vertexInputState.pVertexBindingDescriptions = vertexInputBindings.data();
		vertexInputState.vertexAttributeDescriptionCount = static_cast<uint32_t>(vertexInputAttributes.size());
		vertexInputState.pVertexAttributeDescriptions = vertexInputAttributes.data();

		pipelineCreateInfo.pVertexInputState = &vertexInputState;

		// Create the graphics pipeline state objects

		// We are using this pipeline as the base for the other pipelines (derivatives)
		// Pipeline derivatives can be used for pipelines that share most of their state
		// Depending on the implementation this may result in better performance for pipeline 
		// switchting and faster creation time
		pipelineCreateInfo.flags = vk::PipelineCreateFlagBits::eAllowDerivatives;

		// Textured pipeline
		// Phong shading pipeline
		shaderStages[0] = loadShader(getAssetPath() + "shaders/phong.vert.spv", vk::ShaderStageFlagBits::eVertex);
		shaderStages[1] = loadShader(getAssetPath() + "shaders/phong.frag.spv", vk::ShaderStageFlagBits::eFragment);
		pipelines.phong = device.createGraphicsPipelines(pipelineCache, pipelineCreateInfo)[0];

		// All pipelines created after the base pipeline will be derivatives
		pipelineCreateInfo.flags = vk::PipelineCreateFlagBits::eDerivative;
		// Base pipeline will be our first created pipeline
		pipelineCreateInfo.basePipelineHandle = pipelines.phong;
		// It's only allowed to either use a handle or index for the base pipeline
		// As we use the handle, we must set the index to -1 (see section 9.5 of the specification)
		pipelineCreateInfo.basePipelineIndex = -1;

		// Toon shading pipeline
		shaderStages[0] = loadShader(getAssetPath() + "shaders/toon.vert.spv", vk::ShaderStageFlagBits::eVertex);
		shaderStages[1] = loadShader(getAssetPath() + "shaders/toon.frag.spv", vk::ShaderStageFlagBits::eFragment);
		pipelines.toon = device.createGraphicsPipelines(pipelineCache, pipelineCreateInfo)[0];

		// Pipeline for wire frame rendering
		// Non solid rendering is not a mandatory Vulkan feature
		if (deviceFeatures.fillModeNonSolid)
		{
			rasterizationState.polygonMode = vk::PolygonMode::eLine;
			shaderStages[0] = loadShader(getAssetPath() + "shaders/wireframe.vert.spv", vk::ShaderStageFlagBits::eVertex);
			shaderStages[1] = loadShader(getAssetPath() + "shaders/wireframe.frag.spv", vk::ShaderStageFlagBits::eFragment);
			pipelines.wireframe = device.createGraphicsPipelines(pipelineCache, pipelineCreateInfo)[0];
		}
	}

	// Prepare and initialize uniform buffer containing shader uniforms
	void prepareUniformBuffers()
	{
		// Create the vertex shader uniform buffer block
		vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eUniformBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			&uniformBuffer,
			sizeof(uboVS));

		// Map persistent
		uniformBuffer.map();

		updateUniformBuffers();
	}

	void updateUniformBuffers()
	{
		uboVS.projection = glm::perspective(glm::radians(60.0f), (float)(width / 3.0f) / (float)height, 0.1f, 256.0f);

		glm::mat4 viewMatrix = glm::translate(glm::mat4(), glm::vec3(0.0f, 0.0f, zoom));

		uboVS.modelView = viewMatrix * glm::translate(glm::mat4(), cameraPos);
		uboVS.modelView = glm::rotate(uboVS.modelView, glm::radians(rotation.x), glm::vec3(1.0f, 0.0f, 0.0f));
		uboVS.modelView = glm::rotate(uboVS.modelView, glm::radians(rotation.y), glm::vec3(0.0f, 1.0f, 0.0f));
		uboVS.modelView = glm::rotate(uboVS.modelView, glm::radians(rotation.z), glm::vec3(0.0f, 0.0f, 1.0f));

		memcpy(uniformBuffer.mapped, &uboVS, sizeof(uboVS));
	}

	void draw()
	{
		VulkanExampleBase::prepareFrame();

		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &drawCmdBuffers[currentBuffer];
		queue.submit(submitInfo, vk::Fence(nullptr));

		VulkanExampleBase::submitFrame();
	}

	void prepare()
	{
		VulkanExampleBase::prepare();
		loadAssets();
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
	}

	virtual void viewChanged()
	{
		updateUniformBuffers();
	}

	virtual void getOverlayText(VulkanTextOverlay *textOverlay)
	{
		textOverlay->addText("Phong shading pipeline",(float)width / 6.0f, height - 35.0f, VulkanTextOverlay::alignCenter);
		textOverlay->addText("Toon shading pipeline", (float)width / 2.0f, height - 35.0f, VulkanTextOverlay::alignCenter);
		textOverlay->addText("Wireframe pipeline", width - (float)width / 6.5f, height - 35.0f, VulkanTextOverlay::alignCenter);
		if (!deviceFeatures.fillModeNonSolid) {
			textOverlay->addText("Non solid fill modes not supported!", width - (float)width / 6.5f, (float)height / 2.0f - 7.5f, VulkanTextOverlay::alignCenter);
		}
	}
};

VULKAN_EXAMPLE_MAIN()