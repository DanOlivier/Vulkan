/*
* Vulkan Example - Displacement mapping with tessellation shaders
*
* Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include "VulkanExampleBase.hpp"
#include "VulkanTexture.hpp"
#include "VulkanModel.hpp"

#include <sstream>
#include <iomanip>

#define VERTEX_BUFFER_BIND_ID 0
#define ENABLE_VALIDATION false

class VulkanExample : public VulkanExampleBase
{
private:
	struct {
		vks::Texture2D colorHeightMap;
	} textures;
public:
	bool splitScreen = true;
	bool displacement = true;

	struct {
		vk::PipelineVertexInputStateCreateInfo inputState;
		std::vector<vk::VertexInputBindingDescription> bindingDescriptions;
		std::vector<vk::VertexInputAttributeDescription> attributeDescriptions;
	} vertices;

	// Vertex layout for the models
	vks::VertexLayout vertexLayout = vks::VertexLayout({
		vks::VERTEX_COMPONENT_POSITION,
		vks::VERTEX_COMPONENT_NORMAL,
		vks::VERTEX_COMPONENT_UV,
	});

	struct {
		vks::Model object;
	} models;

	struct {
		vks::Buffer tessControl, tessEval;
	} uniformBuffers;

	struct UBOTessControl {
		float tessLevel = 64.0f;
	} uboTessControl;

	struct UBOTessEval {
		glm::mat4 projection;
		glm::mat4 model;
		glm::vec4 lightPos = glm::vec4(0.0f, -1.0f, 0.0f, 0.0f);
		float tessAlpha = 1.0f;
		float tessStrength = 0.1f;
	} uboTessEval;

	struct Pipelines {
		vk::Pipeline solid;
		vk::Pipeline wireframe;
	} pipelines;

	vk::PipelineLayout pipelineLayout;
	vk::DescriptorSet descriptorSet;
	vk::DescriptorSetLayout descriptorSetLayout;

	VulkanExample() : VulkanExampleBase(ENABLE_VALIDATION)
	{
		zoom = -1.25f;
		rotation = glm::vec3(-20.0f, 45.0f, 0.0f);
		enableTextOverlay = true;
		title = "Vulkan Example - Tessellation shader displacement mapping";
	}

	~VulkanExample()
	{
		// Clean up used Vulkan resources 
		// Note : Inherited destructor cleans up resources stored in base class
		device.destroyPipeline(pipelines.solid);
		if (pipelines.wireframe) {
			device.destroyPipeline(pipelines.wireframe);
		};

		device.destroyPipelineLayout(pipelineLayout);
		device.destroyDescriptorSetLayout(descriptorSetLayout);

		uniformBuffers.tessControl.destroy();
		uniformBuffers.tessEval.destroy();
		models.object.destroy();
		textures.colorHeightMap.destroy();
	}

	// Enable physical device features required for this example				
	virtual void getEnabledFeatures()
	{
		// Tessellation shader support is required for this example
		if (deviceFeatures.tessellationShader) {
			enabledFeatures.tessellationShader = VK_TRUE;
		}
		else {
			vks::tools::exitFatal("Selected GPU does not support tessellation shaders!", "Feature not supported");
		}
		// Fill mode non solid is required for wireframe display
		if (deviceFeatures.fillModeNonSolid) {
			enabledFeatures.fillModeNonSolid = VK_TRUE;
		}
		else {
			splitScreen = false;
		}
	}

	void loadAssets()
	{
		models.object.loadFromFile(getAssetPath() + "models/plane.obj", vertexLayout, 0.25f, vulkanDevice, queue);
		// Textures
		if (vulkanDevice->features.textureCompressionBC) {
			textures.colorHeightMap.loadFromFile(getAssetPath() + "textures/stonefloor03_color_bc3_unorm.ktx", vk::Format::eBc3UnormBlock, vulkanDevice, queue);
		}
		else if (vulkanDevice->features.textureCompressionASTC_LDR) {
			textures.colorHeightMap.loadFromFile(getAssetPath() + "textures/stonefloor03_color_astc_8x8_unorm.ktx", vk::Format::eAstc8x8UnormBlock, vulkanDevice, queue);
		}
		else if (vulkanDevice->features.textureCompressionETC2) {
			textures.colorHeightMap.loadFromFile(getAssetPath() + "textures/stonefloor03_color_etc2_unorm.ktx", vk::Format::eEtc2R8G8B8UnormBlock, vulkanDevice, queue);
		}
		else {
			vks::tools::exitFatal("Device does not support any compressed texture format!", "Error");
		}
	}

	void reBuildCommandBuffers()
	{
		if (!checkCommandBuffers())
		{
			destroyCommandBuffers();
			createCommandBuffers();
		}
		buildCommandBuffers();
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
			// Set target frame buffer
			renderPassBeginInfo.framebuffer = frameBuffers[i];

			drawCmdBuffers[i].begin(cmdBufInfo);

			drawCmdBuffers[i].beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);

			vk::Viewport viewport = vks::initializers::viewport((float)width, (float)height, 0.0f, 1.0f);
			drawCmdBuffers[i].setViewport(0, viewport);

			vk::Rect2D scissor = vks::initializers::rect2D(splitScreen ? width / 2 : width, height, 0, 0);
			drawCmdBuffers[i].setScissor(0, scissor);

			vkCmdSetLineWidth(drawCmdBuffers[i], 1.0f);

			drawCmdBuffers[i].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSet, nullptr);

			std::vector<vk::DeviceSize> offsets = { 0 };
			drawCmdBuffers[i].bindVertexBuffers(VERTEX_BUFFER_BIND_ID, models.object.vertices.buffer, offsets);
			drawCmdBuffers[i].bindIndexBuffer(models.object.indices.buffer, 0, vk::IndexType::eUint32);

			if (splitScreen)
			{
				drawCmdBuffers[i].bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.wireframe);
				drawCmdBuffers[i].drawIndexed(models.object.indexCount, 1, 0, 0, 0);
				scissor.offset.x = width / 2;
				drawCmdBuffers[i].setScissor(0, scissor);
			}

			drawCmdBuffers[i].bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.solid);
			drawCmdBuffers[i].drawIndexed(models.object.indexCount, 1, 0, 0, 0);

			drawCmdBuffers[i].endRenderPass();

			drawCmdBuffers[i].end();
		}
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

		// Location 1 : Normals
		vertices.attributeDescriptions[1] =
			vks::initializers::vertexInputAttributeDescription(
				VERTEX_BUFFER_BIND_ID,
				1,
				vk::Format::eR32G32B32Sfloat,
				sizeof(float) * 3);

		// Location 2 : Texture coordinates
		vertices.attributeDescriptions[2] =
			vks::initializers::vertexInputAttributeDescription(
				VERTEX_BUFFER_BIND_ID,
				2,
				vk::Format::eR32G32Sfloat,
				sizeof(float) * 6);

		vertices.inputState = vks::initializers::pipelineVertexInputStateCreateInfo();
		vertices.inputState.vertexBindingDescriptionCount = static_cast<uint32_t>(vertices.bindingDescriptions.size());
		vertices.inputState.pVertexBindingDescriptions = vertices.bindingDescriptions.data();
		vertices.inputState.vertexAttributeDescriptionCount = static_cast<uint32_t>(vertices.attributeDescriptions.size());
		vertices.inputState.pVertexAttributeDescriptions = vertices.attributeDescriptions.data();
	}

	void setupDescriptorPool()
	{
		// Example uses two ubos and two image samplers
		std::vector<vk::DescriptorPoolSize> poolSizes =
		{
			vks::initializers::descriptorPoolSize(vk::DescriptorType::eUniformBuffer, 2),
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
			// Binding 0 : Tessellation control shader ubo
			vks::initializers::descriptorSetLayoutBinding(
				vk::DescriptorType::eUniformBuffer, 
				vk::ShaderStageFlagBits::eTessellationControl, 
				0),
			// Binding 1 : Tessellation evaluation shader ubo
			vks::initializers::descriptorSetLayoutBinding(
				vk::DescriptorType::eUniformBuffer,
				vk::ShaderStageFlagBits::eTessellationEvaluation, 
				1),
			// Binding 2 : Combined color (rgb) and height (alpha) map
			vks::initializers::descriptorSetLayoutBinding(
				vk::DescriptorType::eCombinedImageSampler,
				vk::ShaderStageFlagBits::eTessellationEvaluation | vk::ShaderStageFlagBits::eFragment,
				2),
		};

		vk::DescriptorSetLayoutCreateInfo descriptorLayout =
			vks::initializers::descriptorSetLayoutCreateInfo(
				setLayoutBindings);

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

		// Color and height map image descriptor
		vk::DescriptorImageInfo texDescriptor =
			vks::initializers::descriptorImageInfo(
				textures.colorHeightMap.sampler,
				textures.colorHeightMap.view,
				vk::ImageLayout::eGeneral);

		std::vector<vk::WriteDescriptorSet> writeDescriptorSets =
		{
			// Binding 0 : Tessellation control shader ubo
			vks::initializers::writeDescriptorSet(
				descriptorSet, 
				vk::DescriptorType::eUniformBuffer, 
				0, 
				&uniformBuffers.tessControl.descriptor),
			// Binding 1 : Tessellation evaluation shader ubo
			vks::initializers::writeDescriptorSet(
				descriptorSet, 
				vk::DescriptorType::eUniformBuffer,
				1, 
				&uniformBuffers.tessEval.descriptor),
			// Binding 2 : Color and displacement map (alpha channel)
			vks::initializers::writeDescriptorSet(
				descriptorSet,
				vk::DescriptorType::eCombinedImageSampler,
				2,
				&texDescriptor),
		};

		device.updateDescriptorSets(writeDescriptorSets, nullptr);
	}

	void preparePipelines()
	{
		vk::PipelineInputAssemblyStateCreateInfo inputAssemblyState =
			vks::initializers::pipelineInputAssemblyStateCreateInfo(
				vk::PrimitiveTopology::ePatchList,
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
			vk::DynamicState::eScissor,
			vk::DynamicState::eLineWidth
		};
		vk::PipelineDynamicStateCreateInfo dynamicState =
			vks::initializers::pipelineDynamicStateCreateInfo(
				dynamicStateEnables);

		vk::PipelineTessellationStateCreateInfo tessellationState =
			vks::initializers::pipelineTessellationStateCreateInfo(3);

		// Tessellation pipeline
		// Load shaders
		std::array<vk::PipelineShaderStageCreateInfo, 4> shaderStages;
		shaderStages[0] = loadShader(getAssetPath() + "shaders/displacement/base.vert.spv", vk::ShaderStageFlagBits::eVertex);
		shaderStages[1] = loadShader(getAssetPath() + "shaders/displacement/base.frag.spv", vk::ShaderStageFlagBits::eFragment);
		shaderStages[2] = loadShader(getAssetPath() + "shaders/displacement/displacement.tesc.spv", vk::ShaderStageFlagBits::eTessellationControl);
		shaderStages[3] = loadShader(getAssetPath() + "shaders/displacement/displacement.tese.spv", vk::ShaderStageFlagBits::eTessellationEvaluation);

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
		pipelineCreateInfo.pTessellationState = &tessellationState;
		pipelineCreateInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
		pipelineCreateInfo.pStages = shaderStages.data();
		pipelineCreateInfo.renderPass = renderPass;

		// Solid pipeline
		rasterizationState.cullMode = vk::CullModeFlagBits::eBack;
		pipelines.solid = device.createGraphicsPipelines(pipelineCache, pipelineCreateInfo)[0];
		if (deviceFeatures.fillModeNonSolid) {
			// Wireframe pipeline
			rasterizationState.polygonMode = vk::PolygonMode::eLine;
			rasterizationState.cullMode = vk::CullModeFlagBits::eNone;
			pipelines.wireframe = device.createGraphicsPipelines(pipelineCache, pipelineCreateInfo)[0];
		}
	}

	// Prepare and initialize uniform buffer containing shader uniforms
	void prepareUniformBuffers()
	{
		// Tessellation evaluation shader uniform buffer
		vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eUniformBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			&uniformBuffers.tessEval,
			sizeof(uboTessEval));

		// Tessellation control shader uniform buffer
		vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eUniformBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			&uniformBuffers.tessControl,
			sizeof(uboTessControl));

		// Map persistent
		uniformBuffers.tessControl.map();
		uniformBuffers.tessEval.map();

		updateUniformBuffers();
	}

	void updateUniformBuffers()
	{
		// Tessellation eval
		glm::mat4 viewMatrix = glm::mat4();
		uboTessEval.projection = glm::perspective(glm::radians(45.0f), (float)(width) / (float)height, 0.1f, 256.0f);
		viewMatrix = glm::translate(viewMatrix, glm::vec3(0.0f, 0.0f, zoom));

		uboTessEval.model = glm::mat4();
		uboTessEval.model = viewMatrix * glm::translate(uboTessEval.model, glm::vec3(0, 0, 0));
		uboTessEval.model = glm::rotate(uboTessEval.model, glm::radians(rotation.x), glm::vec3(1.0f, 0.0f, 0.0f));
		uboTessEval.model = glm::rotate(uboTessEval.model, glm::radians(rotation.y), glm::vec3(0.0f, 1.0f, 0.0f));
		uboTessEval.model = glm::rotate(uboTessEval.model, glm::radians(rotation.z), glm::vec3(0.0f, 0.0f, 1.0f));

		uboTessEval.lightPos.y = -0.5f - uboTessEval.tessStrength;

		memcpy(uniformBuffers.tessEval.mapped, &uboTessEval, sizeof(uboTessEval));

		// Tessellation control
		float savedLevel = uboTessControl.tessLevel;
		if (!displacement)
		{
			uboTessControl.tessLevel = 1.0f;
		}

		memcpy(uniformBuffers.tessControl.mapped, &uboTessControl, sizeof(uboTessControl));

		if (!displacement)
		{
			uboTessControl.tessLevel = savedLevel;
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

	void prepare()
	{
		// Check if device supports tessellation shaders
		if (!deviceFeatures.tessellationShader)
		{
			vks::tools::exitFatal("Selected GPU does not support tessellation shaders!", "Feature not supported");
		}

		VulkanExampleBase::prepare();
		loadAssets();
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
	}

	virtual void viewChanged()
	{
		updateUniformBuffers();
	}

	void changeTessellationLevel(float delta)
	{
		uboTessControl.tessLevel += delta;
		uboTessControl.tessLevel = fmax(1.0f, fmin(uboTessControl.tessLevel, 32.0f));
		updateUniformBuffers();
		updateTextOverlay();
	}

	void changeTessellationStrength(float delta)
	{
		uboTessEval.tessStrength += delta;
		uboTessEval.tessStrength = fmax(0.0f, fmin(uboTessEval.tessStrength, 1.0f));
		updateUniformBuffers();
		updateTextOverlay();
	}

	void toggleSplitScreen()
	{
		splitScreen = !splitScreen;
		reBuildCommandBuffers();
		updateUniformBuffers();
	}

	void toggleDisplacement()
	{
		displacement = !displacement;
		updateUniformBuffers();
	}

	virtual void keyPressed(uint32_t keyCode)
	{
		switch (keyCode)
		{
		case KEY_KPADD:
		case GAMEPAD_BUTTON_R1:
			changeTessellationStrength(0.025f);
			break;
		case KEY_KPSUB:
		case GAMEPAD_BUTTON_L1:
			changeTessellationStrength(-0.025f);
			break;
		case KEY_D:
		case GAMEPAD_BUTTON_A:
			toggleDisplacement();
			break;
		case KEY_S:
		case GAMEPAD_BUTTON_X:
			if (deviceFeatures.fillModeNonSolid) {
				toggleSplitScreen();
			};
			break;
		}
	}

	virtual void getOverlayText(VulkanTextOverlay *textOverlay)
	{
		std::stringstream ss;
		ss << std::setprecision(2) << std::fixed << uboTessEval.tessStrength;
#if defined(__ANDROID__)
		textOverlay->addText("Tessellation height: " + ss.str() + " (Buttons L1/R1)", 5.0f, 85.0f, VulkanTextOverlay::alignLeft);
		textOverlay->addText("\"Button A\" to toggle displacement", 5.0f, 100.0f, VulkanTextOverlay::alignLeft);
		if (deviceFeatures.fillModeNonSolid) {
			textOverlay->addText("\"Button X\" to toggle splitscreen", 5.0f, 115.0f, VulkanTextOverlay::alignLeft);
		}
#else
		textOverlay->addText("Tessellation height: " + ss.str() + " (numpad +/-)", 5.0f, 85.0f, VulkanTextOverlay::alignLeft);
		textOverlay->addText("\"d\" to toggle displacement", 5.0f, 100.0f, VulkanTextOverlay::alignLeft);
		if (deviceFeatures.fillModeNonSolid) {
			textOverlay->addText("\"s\" to toggle splitscreen", 5.0f, 115.0f, VulkanTextOverlay::alignLeft);
		}
#endif
	}
};

VULKAN_EXAMPLE_MAIN()