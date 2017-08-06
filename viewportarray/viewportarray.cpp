/*
* Vulkan Example - Viewport array with single pass rendering using geometry shaders
*
* Copyright (C) 2017 by Sascha Willems - www.saschawillems.de
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
#include "VulkanModel.hpp"

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

	vks::Model scene;

	struct UBOGS {
		glm::mat4 projection[2];
		glm::mat4 modelview[2];
		glm::vec4 lightPos = glm::vec4(-2.5f, -3.5f, 0.0f, 1.0f);
	} uboGS;

	vks::Buffer uniformBufferGS;

	vk::Pipeline pipeline;
	vk::PipelineLayout pipelineLayout;
	vk::DescriptorSet descriptorSet;
	vk::DescriptorSetLayout descriptorSetLayout;

	// Camera and view properties
	float eyeSeparation = 0.08f;
	const float focalLength = 0.5f;
	const float fov = 90.0f;
	const float zNear = 0.1f;
	const float zFar = 256.0f;

	VulkanExample() : VulkanExampleBase(ENABLE_VALIDATION)
	{
		title = "Vulkan Example - Viewport arrays";
		enableTextOverlay = true;
		camera.type = Camera::CameraType::firstperson;
		camera.setRotation(glm::vec3(0.0f, 90.0f, 0.0f));
		camera.setTranslation(glm::vec3(7.0f, 3.2f, 0.0f));
		camera.movementSpeed = 5.0f;
	}

	~VulkanExample()
	{
		device.destroyPipeline(pipeline);

		device.destroyPipelineLayout(pipelineLayout);
		device.destroyDescriptorSetLayout(descriptorSetLayout);

		scene.destroy();

		uniformBufferGS.destroy();
	}

	// Enable physical device features required for this example				
	virtual void getEnabledFeatures()
	{
		// Geometry shader support is required for this example
		if (deviceFeatures.geometryShader) {
			enabledFeatures.geometryShader = VK_TRUE;
		}
		else {
			vks::tools::exitFatal("Selected GPU does not support geometry shaders!", "Feature not supported");
		}
		// Multiple viewports must be supported
		if (deviceFeatures.multiViewport) {
			enabledFeatures.multiViewport = VK_TRUE;
		}
		else {
			vks::tools::exitFatal("Selected GPU does not support multi viewports!", "Feature not supported");
		}
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

			drawCmdBuffers[i].beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);

			vk::Viewport viewports[2];
			// Left
			viewports[0] = { 0, 0, (float)width / 2.0f, (float)height, 0.0, 1.0f };
			// Right
			viewports[1] = { (float)width / 2.0f, 0, (float)width / 2.0f, (float)height, 0.0, 1.0f };

			vkCmdSetViewport(drawCmdBuffers[i], 0, 2, viewports);

			vk::Rect2D scissorRects[2] = {
				vks::initializers::rect2D(width/2, height, 0, 0),
				vks::initializers::rect2D(width/2, height, width / 2, 0),
			};
			vkCmdSetScissor(drawCmdBuffers[i], 0, 2, scissorRects);

			vkCmdSetLineWidth(drawCmdBuffers[i], 1.0f);

			drawCmdBuffers[i].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSet, nullptr);

			vk::DeviceSize offsets[1] = { 0 };
			drawCmdBuffers[i].bindVertexBuffers(0, 1, scene.vertices.buffer, offsets);
			drawCmdBuffers[i].bindIndexBuffer(scene.indices.buffer, 0, vk::IndexType::eUint32);
			drawCmdBuffers[i].bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);
			drawCmdBuffers[i].drawIndexed(scene.indexCount, 1, 0, 0, 0);

			drawCmdBuffers[i].endRenderPass();

			drawCmdBuffers[i].end();
		}
	}

	void loadAssets()
	{
		scene.loadFromFile(getAssetPath() + "models/sampleroom.dae", vertexLayout, 0.25f, vulkanDevice, queue);		
	}

	void setupDescriptorPool()
	{
		// Example uses two ubos
		std::vector<vk::DescriptorPoolSize> poolSizes = {
			vks::initializers::descriptorPoolSize(vk::DescriptorType::eUniformBuffer, 1),
		};

		vk::DescriptorPoolCreateInfo descriptorPoolInfo =
			vks::initializers::descriptorPoolCreateInfo(static_cast<uint32_t>(poolSizes.size()), poolSizes.data(), 1);

		descriptorPool = device.createDescriptorPool(descriptorPoolInfo);
	}

	void setupDescriptorSetLayout()
	{
		std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings = {			
			vks::initializers::descriptorSetLayoutBinding(vk::DescriptorType::eUniformBuffer, vk::ShaderStageFlagBits::eGeometry, 0)	// Binding 1: Geometry shader ubo
		};

		vk::DescriptorSetLayoutCreateInfo descriptorLayout =
			vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);

		descriptorSetLayout = device.createDescriptorSetLayout(descriptorLayout);

		vk::PipelineLayoutCreateInfo pPipelineLayoutCreateInfo =
			vks::initializers::pipelineLayoutCreateInfo(&descriptorSetLayout, 1);

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
			vks::initializers::writeDescriptorSet(descriptorSet, vk::DescriptorType::eUniformBuffer, 0, &uniformBufferGS.descriptor),	// Binding 0 :Geometry shader ubo
		};

		device.updateDescriptorSets(writeDescriptorSets);
	}

	void preparePipelines()
	{
		vk::PipelineInputAssemblyStateCreateInfo inputAssemblyState =
			vks::initializers::pipelineInputAssemblyStateCreateInfo(vk::PrimitiveTopology::eTriangleList, 0, VK_FALSE);

		vk::PipelineRasterizationStateCreateInfo rasterizationState =
			vks::initializers::pipelineRasterizationStateCreateInfo(vk::PolygonMode::eFill, vk::CullModeFlagBits::eBack, vk::FrontFace::eClockwise);

		vk::PipelineColorBlendAttachmentState blendAttachmentState =
			vks::initializers::pipelineColorBlendAttachmentState(0xf, VK_FALSE);

		vk::PipelineColorBlendStateCreateInfo colorBlendState =
			vks::initializers::pipelineColorBlendStateCreateInfo(1, &blendAttachmentState);

		vk::PipelineDepthStencilStateCreateInfo depthStencilState =
			vks::initializers::pipelineDepthStencilStateCreateInfo(VK_TRUE, VK_TRUE, vk::CompareOp::eLessOrEqual);

		// We use two viewports
		vk::PipelineViewportStateCreateInfo viewportState =
			vks::initializers::pipelineViewportStateCreateInfo(2, 2, 0);

		vk::PipelineMultisampleStateCreateInfo multisampleState =
			vks::initializers::pipelineMultisampleStateCreateInfo(vk::SampleCountFlagBits::e1);

		std::vector<vk::DynamicState> dynamicStateEnables = {
			vk::DynamicState::eViewport,
			vk::DynamicState::eScissor,
			vk::DynamicState::eLineWidth
		};
		vk::PipelineDynamicStateCreateInfo dynamicState =
			vks::initializers::pipelineDynamicStateCreateInfo(dynamicStateEnables);


		// Tessellation pipeline
		// Load shaders
		std::array<vk::PipelineShaderStageCreateInfo, 3> shaderStages;

		vk::GraphicsPipelineCreateInfo pipelineCreateInfo =
			vks::initializers::pipelineCreateInfo(pipelineLayout, renderPass);

		// Vertex bindings an attributes
		std::vector<vk::VertexInputBindingDescription> vertexInputBindings = {
			vks::initializers::vertexInputBindingDescription(0, vertexLayout.stride(), vk::VertexInputRate::eVertex),
		};
		std::vector<vk::VertexInputAttributeDescription> vertexInputAttributes = {
			vks::initializers::vertexInputAttributeDescription(0, 0, vk::Format::eR32G32B32Sfloat, 0),					// Location 0: Position			
			vks::initializers::vertexInputAttributeDescription(0, 1, vk::Format::eR32G32B32Sfloat, sizeof(float) * 3),	// Location 1: Normals			
			vks::initializers::vertexInputAttributeDescription(0, 2, vk::Format::eR32G32B32Sfloat, sizeof(float) * 6),	// Location 2: Color

		};
		vk::PipelineVertexInputStateCreateInfo vertexInputState = vks::initializers::pipelineVertexInputStateCreateInfo();
		vertexInputState.vertexBindingDescriptionCount = static_cast<uint32_t>(vertexInputBindings.size());
		vertexInputState.pVertexBindingDescriptions = vertexInputBindings.data();
		vertexInputState.vertexAttributeDescriptionCount = static_cast<uint32_t>(vertexInputAttributes.size());
		vertexInputState.pVertexAttributeDescriptions = vertexInputAttributes.data();

		pipelineCreateInfo.pVertexInputState = &vertexInputState;
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

		shaderStages[0] = loadShader(getAssetPath() + "shaders/viewportarray/scene.vert.spv", vk::ShaderStageFlagBits::eVertex);
		shaderStages[1] = loadShader(getAssetPath() + "shaders/viewportarray/scene.frag.spv", vk::ShaderStageFlagBits::eFragment);
		// A geometry shader is used to output geometry to multiple viewports in one single pass
		// See the "invoctations" decorator of the layout input in the shader
		shaderStages[2] = loadShader(getAssetPath() + "shaders/viewportarray/multiview.geom.spv", vk::ShaderStageFlagBits::eGeometry);
		pipeline = device.createGraphicsPipelines(pipelineCache, pipelineCreateInfo)[0];
	}

	// Prepare and initialize uniform buffer containing shader uniforms
	void prepareUniformBuffers()
	{
		// Geometry shader uniform buffer block
		VK_CHECK_RESULT(vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eUniformBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			&uniformBufferGS,
			sizeof(uboGS)));

		// Map persistent
		uniformBufferGS.map();

		updateUniformBuffers();
	}

	void updateUniformBuffers()
	{
		// Geometry shader matrices for the two viewports
		// See http://paulbourke.net/stereographics/stereorender/

		// Calculate some variables
		float aspectRatio = (float)(width * 0.5f) / (float)height;
		float wd2 = zNear * tan(glm::radians(fov / 2.0f));
		float ndfl = zNear / focalLength;
		float left, right;
		float top = wd2;
		float bottom = -wd2;

		glm::vec3 camFront;
		camFront.x = -cos(glm::radians(rotation.x)) * sin(glm::radians(rotation.y));
		camFront.y = sin(glm::radians(rotation.x));
		camFront.z = cos(glm::radians(rotation.x)) * cos(glm::radians(rotation.y));
		camFront = glm::normalize(camFront);
		glm::vec3 camRight = glm::normalize(glm::cross(camFront, glm::vec3(0.0f, 1.0f, 0.0f)));

		glm::mat4 rotM = glm::mat4();
		glm::mat4 transM;

		rotM = glm::rotate(rotM, glm::radians(camera.rotation.x), glm::vec3(1.0f, 0.0f, 0.0f));
		rotM = glm::rotate(rotM, glm::radians(camera.rotation.y), glm::vec3(0.0f, 1.0f, 0.0f));
		rotM = glm::rotate(rotM, glm::radians(camera.rotation.z), glm::vec3(0.0f, 0.0f, 1.0f));
	
		// Left eye
		left = -aspectRatio * wd2 + 0.5f * eyeSeparation * ndfl;
		right = aspectRatio * wd2 + 0.5f * eyeSeparation * ndfl;

		transM = glm::translate(glm::mat4(), camera.position - camRight * (eyeSeparation / 2.0f));

		uboGS.projection[0] = glm::frustum(left, right, bottom, top, zNear, zFar);
		uboGS.modelview[0] = rotM * transM;

		// Right eye
		left = -aspectRatio * wd2 - 0.5f * eyeSeparation * ndfl;
		right = aspectRatio * wd2 - 0.5f * eyeSeparation * ndfl;

		transM = glm::translate(glm::mat4(), camera.position + camRight * (eyeSeparation / 2.0f));

		uboGS.projection[1] = glm::frustum(left, right, bottom, top, zNear, zFar);
		uboGS.modelview[1] = rotM * transM;

		memcpy(uniformBufferGS.mapped, &uboGS, sizeof(uboGS));
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

	void changeEyeSeparation(float delta)
	{
		eyeSeparation += delta;
		updateUniformBuffers();
	}

	virtual void keyPressed(uint32_t keyCode)
	{
		switch (keyCode)
		{
		case KEY_KPADD:
		case GAMEPAD_BUTTON_R1:
			changeEyeSeparation(0.005);
			break;
		case KEY_KPSUB:
		case GAMEPAD_BUTTON_L1:
			changeEyeSeparation(-0.005);
			break;
		}
	}

};

VULKAN_EXAMPLE_MAIN()