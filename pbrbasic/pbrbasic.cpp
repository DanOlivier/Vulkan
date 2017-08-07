/*
* Vulkan Example - Physical based shading basics
*
* See http://graphicrants.blogspot.de/2013/08/specular-brdf-reference.html for a good reference to the different functions that make up a specular BRDF
*
* Copyright (C) 2017 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include "vulkanexamplebase.h"
#include "VulkanModel.hpp"

#define VERTEX_BUFFER_BIND_ID 0
#define ENABLE_VALIDATION false
#define GRID_DIM 7
#define OBJ_DIM 0.05f

struct Material {
	// Parameter block used as push constant block
	struct PushBlock {
		float roughness;
		float metallic;
		float r, g, b;
	} params;
	std::string name;
	Material() {};
	Material(std::string n, glm::vec3 c, float r, float m) : name(n) {
		params.roughness = r;
		params.metallic = m;
		params.r = c.r;
		params.g = c.g;
		params.b = c.b;
	};
};

class VulkanExample : public VulkanExampleBase
{
public:
	// Vertex layout for the models
	vks::VertexLayout vertexLayout = vks::VertexLayout({
		vks::VERTEX_COMPONENT_POSITION,
		vks::VERTEX_COMPONENT_NORMAL,
		vks::VERTEX_COMPONENT_UV,
	});

	struct Meshes {
		vks::Model skybox;
		std::vector<vks::Model> objects;
		uint32_t objectIndex = 0;
	} models;

	struct {
		vks::Buffer object;
		vks::Buffer skybox;
		vks::Buffer params;
	} uniformBuffers;

	struct UBOMatrices {
		glm::mat4 projection;
		glm::mat4 model;
		glm::mat4 view;
		glm::vec3 camPos;
	} uboMatrices;

	struct UBOParams {
		glm::vec4 lights[4];
	} uboParams;

	vk::PipelineLayout pipelineLayout;
	vk::Pipeline pipeline;
	vk::DescriptorSetLayout descriptorSetLayout;
	vk::DescriptorSet descriptorSet;

	// Default materials to select from
	std::vector<Material> materials;
	int32_t materialIndex = 0;

	VulkanExample() : VulkanExampleBase(ENABLE_VALIDATION)
	{
		title = "Vulkan Example - Physical based shading basics";
		enableTextOverlay = true;
		camera.type = Camera::CameraType::firstperson;
		camera.setPosition(glm::vec3(10.0f, 13.0f, 1.8f));
		camera.setRotation(glm::vec3(-62.5f, 90.0f, 0.0f));
		camera.movementSpeed = 4.0f;
		camera.setPerspective(60.0f, (float)width / (float)height, 0.1f, 256.0f);
		camera.rotationSpeed = 0.25f;
		paused = true;
		timerSpeed *= 0.25f;

		// Setup some default materials (source: https://seblagarde.wordpress.com/2011/08/17/feeding-a-physical-based-lighting-mode/)
		materials.push_back(Material("Gold", glm::vec3(1.0f, 0.765557f, 0.336057f), 0.1f, 1.0f));
		materials.push_back(Material("Copper", glm::vec3(0.955008f, 0.637427f, 0.538163f), 0.1f, 1.0f));
		materials.push_back(Material("Chromium", glm::vec3(0.549585f, 0.556114f, 0.554256f), 0.1f, 1.0f));
		materials.push_back(Material("Nickel", glm::vec3(0.659777f, 0.608679f, 0.525649f), 0.1f, 1.0f));
		materials.push_back(Material("Titanium", glm::vec3(0.541931f, 0.496791f, 0.449419f), 0.1f, 1.0f));
		materials.push_back(Material("Cobalt", glm::vec3(0.662124f, 0.654864f, 0.633732f), 0.1f, 1.0f));
		materials.push_back(Material("Platinum", glm::vec3(0.672411f, 0.637331f, 0.585456f), 0.1f, 1.0f));
		// Testing materials
		materials.push_back(Material("White", glm::vec3(1.0f), 0.1f, 1.0f));
		materials.push_back(Material("Red", glm::vec3(1.0f, 0.0f, 0.0f), 0.1f, 1.0f));
		materials.push_back(Material("Blue", glm::vec3(0.0f, 0.0f, 1.0f), 0.1f, 1.0f));
		materials.push_back(Material("Black", glm::vec3(0.0f), 0.1f, 1.0f));

		materialIndex = 0;
	}

	~VulkanExample()
	{		
		device.destroyPipeline(pipeline);

		device.destroyPipelineLayout(pipelineLayout);
		device.destroyDescriptorSetLayout(descriptorSetLayout);

		for (auto& model : models.objects) {
			model.destroy();
		}
		models.skybox.destroy();

		uniformBuffers.object.destroy();
		uniformBuffers.skybox.destroy();
		uniformBuffers.params.destroy();
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

			vk::Rect2D scissor = vks::initializers::rect2D(width, height, 0, 0);
			drawCmdBuffers[i].setScissor(0, scissor);

			std::vector<vk::DeviceSize> offsets = { 0 };

			// Objects
			drawCmdBuffers[i].bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);
			drawCmdBuffers[i].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSet, nullptr);
			drawCmdBuffers[i].bindVertexBuffers(VERTEX_BUFFER_BIND_ID, models.objects[models.objectIndex].vertices.buffer, offsets);
			drawCmdBuffers[i].bindIndexBuffer(models.objects[models.objectIndex].indices.buffer, 0, vk::IndexType::eUint32);

			Material mat = materials[materialIndex];

//#define SINGLE_ROW 1	
#ifdef SINGLE_ROW
			mat.params.metallic = 1.0;

			uint32_t objcount = 10;
			for (uint32_t x = 0; x < objcount; x++) {
				glm::vec3 pos = glm::vec3(float(x - (objcount / 2.0f)) * 2.5f, 0.0f, 0.0f);
				mat.params.roughness = glm::clamp((float)x / (float)objcount, 0.005f, 1.0f);
				drawCmdBuffers[i].pushConstants(, pipelineLayout, vk::ShaderStageFlagBits::eVertex, 0, sizeof(glm::vec3), &pos);
				drawCmdBuffers[i].pushConstants(pipelineLayout, vk::ShaderStageFlagBits::eFragment, sizeof(glm::vec3), sizeof(Material::PushBlock), &mat);
				drawCmdBuffers[i].drawIndexed(models.objects[models.objectIndex].indexCount, 1, 0, 0, 0);
			}
#else
			for (uint32_t y = 0; y < GRID_DIM; y++) {
				for (uint32_t x = 0; x < GRID_DIM; x++) {
					glm::vec3 pos = glm::vec3(float(x - (GRID_DIM / 2.0f)) * 2.5f, 0.0f, float(y - (GRID_DIM / 2.0f)) * 2.5f);
					drawCmdBuffers[i].pushConstants(pipelineLayout, vk::ShaderStageFlagBits::eVertex, 0, sizeof(glm::vec3), &pos);
					mat.params.metallic = glm::clamp((float)x / (float)(GRID_DIM - 1), 0.1f, 1.0f);
					mat.params.roughness = glm::clamp((float)y / (float)(GRID_DIM - 1), 0.05f, 1.0f);
					drawCmdBuffers[i].pushConstants(pipelineLayout, vk::ShaderStageFlagBits::eFragment, sizeof(glm::vec3), sizeof(Material::PushBlock), &mat);
					drawCmdBuffers[i].drawIndexed(models.objects[models.objectIndex].indexCount, 1, 0, 0, 0);
				}
			}
#endif
			drawCmdBuffers[i].endRenderPass();

			drawCmdBuffers[i].end();
		}
	}

	void loadAssets()
	{
		// Skybox
		models.skybox.loadFromFile(getAssetPath() + "models/cube.obj", vertexLayout, 1.0f, vulkanDevice, queue);
		// Objects
		std::vector<std::string> filenames = { "geosphere.obj", "teapot.dae", "torusknot.obj", "venus.fbx" };
		for (auto file : filenames) {
			vks::Model model;
			model.loadFromFile(getAssetPath() + "models/" + file, vertexLayout, OBJ_DIM * (file == "venus.fbx" ? 3.0f : 1.0f), vulkanDevice, queue);
			models.objects.push_back(model);
		}
	}

	void setupDescriptorSetLayout()
	{
		std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings = {
			vks::initializers::descriptorSetLayoutBinding(vk::DescriptorType::eUniformBuffer, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0),
			vks::initializers::descriptorSetLayoutBinding(vk::DescriptorType::eUniformBuffer, vk::ShaderStageFlagBits::eFragment, 1),
		};

		vk::DescriptorSetLayoutCreateInfo descriptorLayout =
			vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);

		descriptorSetLayout = device.createDescriptorSetLayout(descriptorLayout);

		vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo =
			vks::initializers::pipelineLayoutCreateInfo(&descriptorSetLayout, 1);

		std::vector<vk::PushConstantRange> pushConstantRanges = {
			vks::initializers::pushConstantRange(vk::ShaderStageFlagBits::eVertex, sizeof(glm::vec3), 0),
			vks::initializers::pushConstantRange(vk::ShaderStageFlagBits::eFragment, sizeof(Material::PushBlock), sizeof(glm::vec3)),
		};

		pipelineLayoutCreateInfo.pushConstantRangeCount = 2;
		pipelineLayoutCreateInfo.pPushConstantRanges = pushConstantRanges.data();

		pipelineLayout = device.createPipelineLayout(pipelineLayoutCreateInfo);
	}

	void setupDescriptorSets()
	{
		// Descriptor Pool
		std::vector<vk::DescriptorPoolSize> poolSizes = {
			vks::initializers::descriptorPoolSize(vk::DescriptorType::eUniformBuffer, 4),
		};

		vk::DescriptorPoolCreateInfo descriptorPoolInfo =
			vks::initializers::descriptorPoolCreateInfo(poolSizes, 2);

		descriptorPool = device.createDescriptorPool(descriptorPoolInfo);

		// Descriptor sets

		vk::DescriptorSetAllocateInfo allocInfo =
			vks::initializers::descriptorSetAllocateInfo(descriptorPool, &descriptorSetLayout, 1);

		// 3D object descriptor set
		descriptorSet = device.allocateDescriptorSets(allocInfo)[0];

		std::vector<vk::WriteDescriptorSet> writeDescriptorSets = {
			vks::initializers::writeDescriptorSet(descriptorSet, vk::DescriptorType::eUniformBuffer, 0, &uniformBuffers.object.descriptor),
			vks::initializers::writeDescriptorSet(descriptorSet, vk::DescriptorType::eUniformBuffer, 1, &uniformBuffers.params.descriptor),
		};
		device.updateDescriptorSets(writeDescriptorSets, nullptr);
	}

	void preparePipelines()
	{
		vk::PipelineInputAssemblyStateCreateInfo inputAssemblyState =
			vks::initializers::pipelineInputAssemblyStateCreateInfo(vk::PrimitiveTopology::eTriangleList, vk::PipelineInputAssemblyStateCreateFlags(), VK_FALSE);

		vk::PipelineRasterizationStateCreateInfo rasterizationState =
			vks::initializers::pipelineRasterizationStateCreateInfo(vk::PolygonMode::eFill, vk::CullModeFlagBits::eBack, vk::FrontFace::eCounterClockwise);

		vk::PipelineColorBlendAttachmentState blendAttachmentState =
			vks::initializers::pipelineColorBlendAttachmentState(
				vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA, 
				VK_FALSE);

		vk::PipelineColorBlendStateCreateInfo colorBlendState =
			vks::initializers::pipelineColorBlendStateCreateInfo(1, &blendAttachmentState);

		vk::PipelineDepthStencilStateCreateInfo depthStencilState =
			vks::initializers::pipelineDepthStencilStateCreateInfo(VK_FALSE, VK_FALSE, vk::CompareOp::eLessOrEqual);

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
		pipelineCreateInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
		pipelineCreateInfo.pStages = shaderStages.data();

		// Vertex bindings an attributes
		// Binding description
		std::vector<vk::VertexInputBindingDescription> vertexInputBindings = {
			vks::initializers::vertexInputBindingDescription(0, vertexLayout.stride(), vk::VertexInputRate::eVertex),
		};

		// Attribute descriptions
		std::vector<vk::VertexInputAttributeDescription> vertexInputAttributes = {
			vks::initializers::vertexInputAttributeDescription(0, 0, vk::Format::eR32G32B32Sfloat, 0),					// Position
			vks::initializers::vertexInputAttributeDescription(0, 1, vk::Format::eR32G32B32Sfloat, sizeof(float) * 3),	// Normal
			vks::initializers::vertexInputAttributeDescription(0, 2, vk::Format::eR32G32Sfloat, sizeof(float) * 5),		// UV
		};

		vk::PipelineVertexInputStateCreateInfo vertexInputState = vks::initializers::pipelineVertexInputStateCreateInfo();
		vertexInputState.vertexBindingDescriptionCount = static_cast<uint32_t>(vertexInputBindings.size());
		vertexInputState.pVertexBindingDescriptions = vertexInputBindings.data();
		vertexInputState.vertexAttributeDescriptionCount = static_cast<uint32_t>(vertexInputAttributes.size());
		vertexInputState.pVertexAttributeDescriptions = vertexInputAttributes.data();

		pipelineCreateInfo.pVertexInputState = &vertexInputState;

		// PBR pipeline
		shaderStages[0] = loadShader(getAssetPath() + "shaders/pbrbasic/pbr.vert.spv", vk::ShaderStageFlagBits::eVertex);
		shaderStages[1] = loadShader(getAssetPath() + "shaders/pbrbasic/pbr.frag.spv", vk::ShaderStageFlagBits::eFragment);
		// Enable depth test and write
		depthStencilState.depthWriteEnable = VK_TRUE;
		depthStencilState.depthTestEnable = VK_TRUE;
		// Flip cull mode
		rasterizationState.cullMode = vk::CullModeFlagBits::eFront;
		pipeline = device.createGraphicsPipelines(pipelineCache, pipelineCreateInfo)[0];
	}

	// Prepare and initialize uniform buffer containing shader uniforms
	void prepareUniformBuffers()
	{
		// Objact vertex shader uniform buffer
		vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eUniformBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			&uniformBuffers.object,
			sizeof(uboMatrices));

		// Skybox vertex shader uniform buffer
		vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eUniformBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			&uniformBuffers.skybox,
			sizeof(uboMatrices));

		// Shared parameter uniform buffer
		vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eUniformBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			&uniformBuffers.params,
			sizeof(uboParams));

		// Map persistent
		uniformBuffers.object.map();
		uniformBuffers.skybox.map();
		uniformBuffers.params.map();

		updateUniformBuffers();
		updateLights();
	}

	void updateUniformBuffers()
	{
		// 3D object
		uboMatrices.projection = camera.matrices.perspective;
		uboMatrices.view = camera.matrices.view;
		uboMatrices.model = glm::rotate(glm::mat4(), glm::radians(-90.0f + (models.objectIndex == 1 ? 45.0f : 0.0f)), glm::vec3(0.0f, 1.0f, 0.0f));
		uboMatrices.camPos = camera.position * -1.0f;
		memcpy(uniformBuffers.object.mapped, &uboMatrices, sizeof(uboMatrices));

		// Skybox
		uboMatrices.model = glm::mat4(glm::mat3(camera.matrices.view));
		memcpy(uniformBuffers.skybox.mapped, &uboMatrices, sizeof(uboMatrices));
	}

	void updateLights()
	{
		const float p = 15.0f;
		uboParams.lights[0] = glm::vec4(-p, -p*0.5f, -p, 1.0f);
		uboParams.lights[1] = glm::vec4(-p, -p*0.5f,  p, 1.0f);
		uboParams.lights[2] = glm::vec4( p, -p*0.5f,  p, 1.0f);
		uboParams.lights[3] = glm::vec4( p, -p*0.5f, -p, 1.0f);

		if (!paused)
		{
			uboParams.lights[0].x = sin(glm::radians(timer * 360.0f)) * 20.0f;
			uboParams.lights[0].z = cos(glm::radians(timer * 360.0f)) * 20.0f;
			uboParams.lights[1].x = cos(glm::radians(timer * 360.0f)) * 20.0f;
			uboParams.lights[1].y = sin(glm::radians(timer * 360.0f)) * 20.0f;
		}

		memcpy(uniformBuffers.params.mapped, &uboParams, sizeof(uboParams));
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
		setupDescriptorSets();
		buildCommandBuffers();
		prepared = true;
	}

	virtual void render()
	{
		if (!prepared)
			return;
		draw();
		if (!paused)
			updateLights();
	}

	virtual void viewChanged()
	{
		updateUniformBuffers();
		updateTextOverlay();
	}

	void toggleObject()
	{
		models.objectIndex++;
		if (models.objectIndex >= static_cast<uint32_t>(models.objects.size()))
		{
			models.objectIndex = 0;
		}
		updateUniformBuffers();
		buildCommandBuffers();
	}

	void toggleMaterial(int32_t dir)
	{
		materialIndex += dir;
		if (materialIndex < 0) {
			materialIndex = static_cast<int32_t>(materials.size()) - 1;
		}
		if (materialIndex > static_cast<int32_t>(materials.size()) - 1) {
			materialIndex = 0;
		}
		buildCommandBuffers();
		updateTextOverlay();
	}

	virtual void keyPressed(uint32_t keyCode)
	{
		switch (keyCode)
		{
		case KEY_SPACE:
		case GAMEPAD_BUTTON_X:
			toggleObject();
			break;
		case KEY_KPADD:
		case GAMEPAD_BUTTON_R1:
			toggleMaterial(1);
			break;
		case KEY_KPSUB:
		case GAMEPAD_BUTTON_L1:
			toggleMaterial(-1);
			break;
		}
	}

	virtual void getOverlayText(VulkanTextOverlay *textOverlay)
	{
#if defined(__ANDROID__)
		textOverlay->addText("Base material: " + materials[materialIndex].name + " (L1/R1)", 5.0f, 85.0f, VulkanTextOverlay::alignLeft);
		textOverlay->addText("\"X\" to toggle object", 5.0f, 100.0f, VulkanTextOverlay::alignLeft);
#else
		textOverlay->addText("Base material: " + materials[materialIndex].name + " (-/+)", 5.0f, 85.0f, VulkanTextOverlay::alignLeft);
		textOverlay->addText("\"space\" to toggle object", 5.0f, 100.0f, VulkanTextOverlay::alignLeft);
#endif
	}
};

VULKAN_EXAMPLE_MAIN()