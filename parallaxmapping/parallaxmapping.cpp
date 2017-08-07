/*
* Vulkan Example - Parallax Mapping
*
* Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include "VulkanExampleBase.hpp"
#include "VulkanTexture.hpp"
#include "VulkanModel.hpp"

#define VERTEX_BUFFER_BIND_ID 0
#define ENABLE_VALIDATION false

class VulkanExample : public VulkanExampleBase
{
public:
	struct {
		vks::Texture2D colorMap;
		// Normals and height are combined into one texture (height = alpha channel)
		vks::Texture2D normalHeightMap;
	} textures;

	// Vertex layout for the models
	vks::VertexLayout vertexLayout = vks::VertexLayout({
		vks::VERTEX_COMPONENT_POSITION,
		vks::VERTEX_COMPONENT_UV,
		vks::VERTEX_COMPONENT_NORMAL,
		vks::VERTEX_COMPONENT_TANGENT,
		vks::VERTEX_COMPONENT_BITANGENT,
	});

	struct {
		vks::Model quad;
	} models;

	struct {
		vks::Buffer vertexShader;
		vks::Buffer fragmentShader;
	} uniformBuffers;

	struct {

		struct {
			glm::mat4 projection;
			glm::mat4 view;
			glm::mat4 model;
			glm::vec4 lightPos = glm::vec4(0.0f, -2.0f, 0.0f, 1.0f);
			glm::vec4 cameraPos;
		} vertexShader;

		struct {
			float heightScale = 0.1f;
			// Basic parallax mapping needs a bias to look any good (and is hard to tweak)
			float parallaxBias = -0.02f;
			// Number of layers for steep parallax and parallax occlusion (more layer = better result for less performance)
			float numLayers = 48.0f;
			// (Parallax) mapping mode to use
			int32_t mappingMode = 4;
		} fragmentShader;

	} ubos;

	vk::PipelineLayout pipelineLayout;
	vk::Pipeline pipeline;
	vk::DescriptorSetLayout descriptorSetLayout;
	vk::DescriptorSet descriptorSet;

	VulkanExample() : VulkanExampleBase(ENABLE_VALIDATION)
	{
		title = "Vulkan Example - Parallax Mapping";
		enableTextOverlay = true;
		timerSpeed *= 0.5f;
		camera.type = Camera::CameraType::firstperson;
		camera.setPosition(glm::vec3(0.0f, 1.25f, 1.5f));
		camera.setRotation(glm::vec3(-45.0f, 180.0f, 0.0f));
		camera.setPerspective(60.0f, (float)width / (float)height, 0.1f, 256.0f);
	}

	~VulkanExample()
	{
		device.destroyPipeline(pipeline);
		
		device.destroyPipelineLayout(pipelineLayout);
		device.destroyDescriptorSetLayout(descriptorSetLayout);

		models.quad.destroy();
			
		uniformBuffers.vertexShader.destroy();
		uniformBuffers.fragmentShader.destroy();

		textures.colorMap.destroy();
		textures.normalHeightMap.destroy();
	}

	void loadAssets()
	{
		models.quad.loadFromFile(getAssetPath() + "models/plane_z.obj", vertexLayout, 0.1f, vulkanDevice, queue);

		// Textures
		textures.normalHeightMap.loadFromFile(getAssetPath() + "textures/rocks_normal_height_rgba.dds", vk::Format::eR8G8B8A8Unorm, vulkanDevice, queue);
		if (vulkanDevice->features.textureCompressionBC) {
			textures.colorMap.loadFromFile(getAssetPath() + "textures/rocks_color_bc3_unorm.dds", vk::Format::eBc3UnormBlock, vulkanDevice, queue);
		}
		else if (vulkanDevice->features.textureCompressionASTC_LDR) {
			textures.colorMap.loadFromFile(getAssetPath() + "textures/rocks_color_astc_8x8_unorm.ktx", vk::Format::eAstc8x8UnormBlock, vulkanDevice, queue);
		}
		else if (vulkanDevice->features.textureCompressionETC2) {
			textures.colorMap.loadFromFile(getAssetPath() + "textures/rocks_color_etc2_unorm.ktx", vk::Format::eEtc2R8G8B8UnormBlock, vulkanDevice, queue);
		}
		else {
			vks::tools::exitFatal("Device does not support any compressed texture format!", "Error");
		}
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

			vk::Rect2D scissor = vks::initializers::rect2D(width, height,	0, 0);
			drawCmdBuffers[i].setScissor(0, scissor);

			drawCmdBuffers[i].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSet, nullptr);

			std::vector<vk::DeviceSize> offsets = { 0 };
			drawCmdBuffers[i].bindVertexBuffers(VERTEX_BUFFER_BIND_ID, models.quad.vertices.buffer, offsets);
			drawCmdBuffers[i].bindIndexBuffer(models.quad.indices.buffer, 0, vk::IndexType::eUint32);

			drawCmdBuffers[i].setViewport(0, viewport);
			drawCmdBuffers[i].bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);
			drawCmdBuffers[i].drawIndexed(models.quad.indexCount, 1, 0, 0, 1);

			drawCmdBuffers[i].endRenderPass();

			drawCmdBuffers[i].end();
		}
	}

	void setupDescriptorPool()
	{
		// Example uses two ubos and two image sampler
		std::vector<vk::DescriptorPoolSize> poolSizes =
		{
			vks::initializers::descriptorPoolSize(vk::DescriptorType::eUniformBuffer, 2),
			vks::initializers::descriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, 2)
		};

		vk::DescriptorPoolCreateInfo descriptorPoolInfo =
			vks::initializers::descriptorPoolCreateInfo(poolSizes, 2);

		descriptorPool = device.createDescriptorPool(descriptorPoolInfo);
	}

	void setupDescriptorSetLayout()
	{
		std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings = {			
			vks::initializers::descriptorSetLayoutBinding(vk::DescriptorType::eUniformBuffer, vk::ShaderStageFlagBits::eVertex, 0),			// Binding 0: Vertex shader uniform buffer		
			vks::initializers::descriptorSetLayoutBinding(vk::DescriptorType::eCombinedImageSampler, vk::ShaderStageFlagBits::eFragment, 1),	// Binding 1: Fragment shader color map image sampler
			vks::initializers::descriptorSetLayoutBinding(vk::DescriptorType::eCombinedImageSampler, vk::ShaderStageFlagBits::eFragment, 2),	// Binding 2: Fragment combined normal and heightmap			
			vks::initializers::descriptorSetLayoutBinding(vk::DescriptorType::eUniformBuffer, vk::ShaderStageFlagBits::eFragment, 3),			// Binding 3: Fragment shader uniform buffer				
		};

		vk::DescriptorSetLayoutCreateInfo descriptorLayout =
			vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);

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
			vks::initializers::writeDescriptorSet(descriptorSet, vk::DescriptorType::eUniformBuffer, 0, &uniformBuffers.vertexShader.descriptor),		// Binding 0: Vertex shader uniform buffer
			vks::initializers::writeDescriptorSet(descriptorSet, vk::DescriptorType::eCombinedImageSampler, 1, &textures.colorMap.descriptor),			// Binding 1: Fragment shader image sampler
			vks::initializers::writeDescriptorSet(descriptorSet, vk::DescriptorType::eCombinedImageSampler, 2, &textures.normalHeightMap.descriptor),	// Binding 2: Combined normal and heightmap
			vks::initializers::writeDescriptorSet(descriptorSet, vk::DescriptorType::eUniformBuffer, 3, &uniformBuffers.fragmentShader.descriptor),		// Binding 3: Fragment shader uniform buffer
		};

		device.updateDescriptorSets(writeDescriptorSets, nullptr);
	}

	void preparePipelines()
	{
		vk::PipelineInputAssemblyStateCreateInfo inputAssemblyState =
			vks::initializers::pipelineInputAssemblyStateCreateInfo(vk::PrimitiveTopology::eTriangleList, vk::PipelineInputAssemblyStateCreateFlags(), VK_FALSE);

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

		// Load shaders
		std::array<vk::PipelineShaderStageCreateInfo, 2> shaderStages;

		vk::GraphicsPipelineCreateInfo pipelineCreateInfo =
			vks::initializers::pipelineCreateInfo(pipelineLayout, renderPass);

		pipelineCreateInfo.pInputAssemblyState = &inputAssemblyState;
		pipelineCreateInfo.pRasterizationState = &rasterizationState;
		pipelineCreateInfo.pColorBlendState = &colorBlendState;
		pipelineCreateInfo.pMultisampleState = &multisampleState;
		pipelineCreateInfo.pViewportState = &viewportState;
		pipelineCreateInfo.pDepthStencilState = &depthStencilState;
		pipelineCreateInfo.pDynamicState = &dynamicState;
		pipelineCreateInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
		pipelineCreateInfo.pStages = shaderStages.data();

		// Vertex bindings an attributes
		std::vector<vk::VertexInputBindingDescription> vertexInputBindings = {
			vks::initializers::vertexInputBindingDescription(0, vertexLayout.stride(), vk::VertexInputRate::eVertex),
		};
		std::vector<vk::VertexInputAttributeDescription> vertexInputAttributes = {			
			vks::initializers::vertexInputAttributeDescription(VERTEX_BUFFER_BIND_ID, 0, vk::Format::eR32G32B32Sfloat, 0),					// Location 0: Position			
			vks::initializers::vertexInputAttributeDescription(VERTEX_BUFFER_BIND_ID, 1, vk::Format::eR32G32Sfloat, sizeof(float) * 3),		// Location 1: Texture coordinates			
			vks::initializers::vertexInputAttributeDescription(VERTEX_BUFFER_BIND_ID, 2, vk::Format::eR32G32B32Sfloat, sizeof(float) * 5),	// Location 2: Normal			
			vks::initializers::vertexInputAttributeDescription(VERTEX_BUFFER_BIND_ID, 3, vk::Format::eR32G32B32Sfloat, sizeof(float) * 8),	// Location 3: Tangent			
			vks::initializers::vertexInputAttributeDescription(VERTEX_BUFFER_BIND_ID, 4, vk::Format::eR32G32B32Sfloat, sizeof(float) * 11),	// Location 4: Bitangent
		};
		vk::PipelineVertexInputStateCreateInfo vertexInputState = vks::initializers::pipelineVertexInputStateCreateInfo();
		vertexInputState.vertexBindingDescriptionCount = static_cast<uint32_t>(vertexInputBindings.size());
		vertexInputState.pVertexBindingDescriptions = vertexInputBindings.data();
		vertexInputState.vertexAttributeDescriptionCount = static_cast<uint32_t>(vertexInputAttributes.size());
		vertexInputState.pVertexAttributeDescriptions = vertexInputAttributes.data();

		pipelineCreateInfo.pVertexInputState = &vertexInputState;

		// Parallax mapping modes pipeline
		shaderStages[0] = loadShader(getAssetPath() + "shaders/parallax/parallax.vert.spv", vk::ShaderStageFlagBits::eVertex);
		shaderStages[1] = loadShader(getAssetPath() + "shaders/parallax/parallax.frag.spv", vk::ShaderStageFlagBits::eFragment);
		pipeline = device.createGraphicsPipelines(pipelineCache, pipelineCreateInfo)[0];
	}

	void prepareUniformBuffers()
	{
		// Vertex shader uniform buffer
		vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eUniformBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			&uniformBuffers.vertexShader,
			sizeof(ubos.vertexShader));

		// Fragment shader uniform buffer
		vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eUniformBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			&uniformBuffers.fragmentShader,
			sizeof(ubos.fragmentShader));

		// Map persistent
		uniformBuffers.vertexShader.map();
		uniformBuffers.fragmentShader.map();

		updateUniformBuffers();
	}

	void updateUniformBuffers()
	{
		// Vertex shader
		ubos.vertexShader.projection = camera.matrices.perspective;
		ubos.vertexShader.view = camera.matrices.view;
		ubos.vertexShader.model = glm::rotate(glm::mat4(), glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));;
		ubos.vertexShader.model = glm::rotate(ubos.vertexShader.model, glm::radians(180.0f), glm::vec3(0.0f, 0.0f, 1.0f));;

		if (!paused) {
			ubos.vertexShader.lightPos.x = sin(glm::radians(timer * 360.0f)) * 1.5f;
			ubos.vertexShader.lightPos.z = cos(glm::radians(timer * 360.0f)) * 1.5f;
		}

		ubos.vertexShader.cameraPos = glm::vec4(camera.position, -1.0f) * -1.0f;

		memcpy(uniformBuffers.vertexShader.mapped, &ubos.vertexShader, sizeof(ubos.vertexShader));

		// Fragment shader
		memcpy(uniformBuffers.fragmentShader.mapped, &ubos.fragmentShader, sizeof(ubos.fragmentShader));
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
		if (!paused)
		{
			updateUniformBuffers();
		}
	}

	virtual void viewChanged()
	{
		updateUniformBuffers();
	}

	void toggleMappingMode()
	{	
		ubos.fragmentShader.mappingMode++;
		if (ubos.fragmentShader.mappingMode > 4) {
			ubos.fragmentShader.mappingMode = 0;
		};
		updateUniformBuffers();
		updateTextOverlay();
	}

	virtual void keyPressed(uint32_t keyCode)
	{
		switch (keyCode)
		{
		case KEY_SPACE:
		case GAMEPAD_BUTTON_A:
		case TOUCH_DOUBLE_TAP:
			toggleMappingMode();
			break;
		}
	}

	virtual void getOverlayText(VulkanTextOverlay *textOverlay)
	{
		const std::vector<std::string> mappingModes = {
			"Color only", "Normal mapping", "Parallax mapping", "Steep parallax mapping", "Parallax occlusion mapping",
		};
#if defined(__ANDROID__)
		textOverlay->addText("Mode: " + mappingModes[ubos.fragmentShader.mappingMode] + " (\"Button A\")", 5.0f, 85.0f, VulkanTextOverlay::alignLeft);
#else		
		textOverlay->addText("Mode: " + mappingModes[ubos.fragmentShader.mappingMode] + " (\"Space\")", 5.0f, 85.0f, VulkanTextOverlay::alignLeft);
#endif
	}
};

VULKAN_EXAMPLE_MAIN()