/*
* Vulkan Example - Shader specialization constants
*
* For details see https://www.khronos.org/registry/vulkan/specs/misc/GL_KHR_vulkan_glsl.txt
*
* Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include "vulkanexamplebase.h"
#include "VulkanTexture.hpp"
#include "VulkanModel.hpp"

#define VERTEX_BUFFER_BIND_ID 0
#define ENABLE_VALIDATION false

class VulkanExample: public VulkanExampleBase 
{
public:
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
		vks::VERTEX_COMPONENT_COLOR,
	});

	struct {
		vks::Model cube;
	} models;

	struct {
		vks::Texture2D colormap;
	} textures;

	vks::Buffer uniformBuffer;

	// Same uniform buffer layout as shader
	struct UBOVS {
		glm::mat4 projection;
		glm::mat4 modelView;
		glm::vec4 lightPos = glm::vec4(0.0f, -2.0f, 1.0f, 0.0f);
	} uboVS;

	vk::PipelineLayout pipelineLayout;
	vk::DescriptorSet descriptorSet;
	vk::DescriptorSetLayout descriptorSetLayout;

	struct {
		vk::Pipeline phong;
		vk::Pipeline toon;
		vk::Pipeline textured;
	} pipelines;

	VulkanExample() : VulkanExampleBase(ENABLE_VALIDATION)
	{
		title = "Vulkan Example - Specialization constants";
		enableTextOverlay = true;
		camera.type = Camera::CameraType::lookat;
		camera.setPerspective(60.0f, ((float)width / 3.0f) / (float)height, 0.1f, 512.0f);
		camera.setRotation(glm::vec3(-40.0f, -90.0f, 0.0f));
		camera.setTranslation(glm::vec3(0.0f, 0.0f, -2.0f));
	}

	~VulkanExample()
	{
		device.destroyPipeline(pipelines.phong);
		device.destroyPipeline(pipelines.textured);
		device.destroyPipeline(pipelines.toon);
		
		device.destroyPipelineLayout(pipelineLayout);
		device.destroyDescriptorSetLayout(descriptorSetLayout);

		models.cube.destroy();
		textures.colormap.destroy();
		uniformBuffer.destroy();
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
			drawCmdBuffers[i].bindVertexBuffers(VERTEX_BUFFER_BIND_ID, models.cube.vertices.buffer, offsets);
			drawCmdBuffers[i].bindIndexBuffer(models.cube.indices.buffer, 0, vk::IndexType::eUint32);

			// Left
			viewport.width = (float)width / 3.0f;
			drawCmdBuffers[i].setViewport(0, viewport);
			drawCmdBuffers[i].bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.phong);
			
			drawCmdBuffers[i].drawIndexed(models.cube.indexCount, 1, 0, 0, 0);

			// Center
			viewport.x = (float)width / 3.0f;
			drawCmdBuffers[i].setViewport(0, viewport);
			drawCmdBuffers[i].bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.toon);
			drawCmdBuffers[i].drawIndexed(models.cube.indexCount, 1, 0, 0, 0);

			// Right
			viewport.x = (float)width / 3.0f + (float)width / 3.0f;
			drawCmdBuffers[i].setViewport(0, viewport);
			drawCmdBuffers[i].bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.textured);
			drawCmdBuffers[i].drawIndexed(models.cube.indexCount, 1, 0, 0, 0);

			drawCmdBuffers[i].endRenderPass();

			drawCmdBuffers[i].end();
		}
	}

	void loadAssets()
	{
		models.cube.loadFromFile(getAssetPath() + "models/color_teapot_spheres.dae", vertexLayout, 0.1f, vulkanDevice, queue);
		textures.colormap.loadFromFile(getAssetPath() + "textures/metalplate_nomips_rgba.ktx", vk::Format::eR8G8B8A8Unorm, vulkanDevice, queue);
	}

	void setupVertexDescriptions()
	{
		// Binding description
		vertices.bindingDescriptions.resize(1);
		vertices.bindingDescriptions = {
			vks::initializers::vertexInputBindingDescription(VERTEX_BUFFER_BIND_ID, vertexLayout.stride(), vk::VertexInputRate::eVertex),
		};

		// Attribute descriptions
		vertices.attributeDescriptions = {
			// Location 0 : Position
			vks::initializers::vertexInputAttributeDescription(
				VERTEX_BUFFER_BIND_ID,
				0,
				vk::Format::eR32G32B32Sfloat,
				0),
			// Location 1 : Color
			vks::initializers::vertexInputAttributeDescription(
				VERTEX_BUFFER_BIND_ID,
				1,
				vk::Format::eR32G32B32Sfloat,
				sizeof(float) * 3),
			// Location 3 : Texture coordinates
			vks::initializers::vertexInputAttributeDescription(
				VERTEX_BUFFER_BIND_ID,
				2,
				vk::Format::eR32G32Sfloat,
				sizeof(float) * 6),
			// Location 2 : Normal
			vks::initializers::vertexInputAttributeDescription(
				VERTEX_BUFFER_BIND_ID,
				3,
				vk::Format::eR32G32B32Sfloat,
				sizeof(float) * 8),
		};

		vertices.inputState = vks::initializers::pipelineVertexInputStateCreateInfo();
		vertices.inputState.vertexBindingDescriptionCount = static_cast<uint32_t>(vertices.bindingDescriptions.size());
		vertices.inputState.pVertexBindingDescriptions = vertices.bindingDescriptions.data();
		vertices.inputState.vertexAttributeDescriptionCount = static_cast<uint32_t>(vertices.attributeDescriptions.size());
		vertices.inputState.pVertexAttributeDescriptions = vertices.attributeDescriptions.data();
	}

	void setupDescriptorPool()
	{
		std::vector<vk::DescriptorPoolSize> poolSizes =
		{
			vks::initializers::descriptorPoolSize(vk::DescriptorType::eUniformBuffer, 1),
			vks::initializers::descriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, 1)
		};

		vk::DescriptorPoolCreateInfo descriptorPoolInfo =
			vks::initializers::descriptorPoolCreateInfo(
				static_cast<uint32_t>(poolSizes.size()),
				poolSizes.data(),
				1);

		descriptorPool = device.createDescriptorPool(descriptorPoolInfo);
	}

	void setupDescriptorSetLayout()
	{
		std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings ={			
			vks::initializers::descriptorSetLayoutBinding(vk::DescriptorType::eUniformBuffer, vk::ShaderStageFlagBits::eVertex, 0),
			vks::initializers::descriptorSetLayoutBinding(vk::DescriptorType::eCombinedImageSampler, vk::ShaderStageFlagBits::eFragment, 1),
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

		std::vector<vk::WriteDescriptorSet> writeDescriptorSets = {			
			vks::initializers::writeDescriptorSet(descriptorSet, vk::DescriptorType::eUniformBuffer, 0, &uniformBuffer.descriptor),
			vks::initializers::writeDescriptorSet(descriptorSet, vk::DescriptorType::eCombinedImageSampler, 1, &textures.colormap.descriptor),
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
				vk::FrontFace::eClockwise);

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
			vk::DynamicState::eLineWidth,
		};
		vk::PipelineDynamicStateCreateInfo dynamicState =
			vks::initializers::pipelineDynamicStateCreateInfo(
				dynamicStateEnables);

		std::array<vk::PipelineShaderStageCreateInfo, 2> shaderStages;

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

		// Prepare specialization data

		// Host data to take specialization constants from
		struct SpecializationData {
			// Sets the lighting model used in the fragment "uber" shader
			uint32_t lightingModel;
			// Parameter for the toon shading part of the fragment shader
			float toonDesaturationFactor = 0.5f;
		} specializationData;

		// Each shader constant of a shader stage corresponds to one map entry
		std::array<vk::SpecializationMapEntry, 2> specializationMapEntries;
		// Shader bindings based on specialization constants are marked by the new "constant_id" layout qualifier:
		//	layout (constant_id = 0) const int LIGHTING_MODEL = 0;
		//	layout (constant_id = 1) const float PARAM_TOON_DESATURATION = 0.0f;

		// Map entry for the lighting model to be used by the fragment shader
		specializationMapEntries[0].constantID = 0;
		specializationMapEntries[0].size = sizeof(specializationData.lightingModel);
		specializationMapEntries[0].offset = 0;

		// Map entry for the toon shader parameter
		specializationMapEntries[1].constantID = 1;
		specializationMapEntries[1].size = sizeof(specializationData.toonDesaturationFactor);
		specializationMapEntries[1].offset = offsetof(SpecializationData, toonDesaturationFactor);

		// Prepare specialization info block for the shader stage
		vk::SpecializationInfo specializationInfo{};
		specializationInfo.dataSize = sizeof(specializationData);
		specializationInfo.mapEntryCount = static_cast<uint32_t>(specializationMapEntries.size());
		specializationInfo.pMapEntries = specializationMapEntries.data();
		specializationInfo.pData = &specializationData;

		// Create pipelines
		// All pipelines will use the same "uber" shader and specialization constants to change branching and parameters of that shader
		shaderStages[0] = loadShader(getAssetPath() + "shaders/specializationconstants/uber.vert.spv", vk::ShaderStageFlagBits::eVertex);
		shaderStages[1] = loadShader(getAssetPath() + "shaders/specializationconstants/uber.frag.spv", vk::ShaderStageFlagBits::eFragment);
		// Specialization info is assigned is part of the shader stage (modul) and must be set after creating the module and before creating the pipeline
		shaderStages[1].pSpecializationInfo = &specializationInfo;

		// Solid phong shading
		specializationData.lightingModel = 0;
		pipelines.phong = device.createGraphicsPipelines(pipelineCache, pipelineCreateInfo)[0];

		// Phong and textured
		specializationData.lightingModel = 1;
		pipelines.toon = device.createGraphicsPipelines(pipelineCache, pipelineCreateInfo)[0];

		// Textured discard
		specializationData.lightingModel = 2;
		pipelines.textured = device.createGraphicsPipelines(pipelineCache, pipelineCreateInfo)[0];
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
		uboVS.projection = camera.matrices.perspective;
		uboVS.modelView = camera.matrices.view;

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
};

VULKAN_EXAMPLE_MAIN()