/*
* Vulkan Example - Deferred shading with shadows from multiple light sources using geometry shader instancing
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
#include <algorithm>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <vulkan/vulkan.hpp>
#include "vulkanexamplebase.h"
#include "VulkanBuffer.hpp"
#include "VulkanFrameBuffer.hpp"
#include "VulkanTexture.hpp"
#include "VulkanModel.hpp"

#define VERTEX_BUFFER_BIND_ID 0
#define ENABLE_VALIDATION false

// Shadowmap properties
#if defined(__ANDROID__)
#define SHADOWMAP_DIM 1024
#else
#define SHADOWMAP_DIM 2048
#endif
// 16 bits of depth is enough for such a small scene
#define SHADOWMAP_FORMAT vk::Format::eD32SfloatS8Uint

#if defined(__ANDROID__)
// Use max. screen dimension as deferred framebuffer size
#define FB_DIM std::max(width,height) 
#else
#define FB_DIM 2048
#endif

// Must match the LIGHT_COUNT define in the shadow and deferred shaders
#define LIGHT_COUNT 3

class VulkanExample : public VulkanExampleBase
{
public:
	bool debugDisplay = false;
	bool enableShadows = true;

	// Keep depth range as small as possible
	// for better shadow map precision
	float zNear = 0.1f;
	float zFar = 64.0f;
	float lightFOV = 100.0f;

	// Depth bias (and slope) are used to avoid shadowing artefacts
	float depthBiasConstant = 1.25f;
	float depthBiasSlope = 1.75f;

	struct {
		struct {
			vks::Texture2D colorMap;
			vks::Texture2D normalMap;
		} model;
		struct {
			vks::Texture2D colorMap;
			vks::Texture2D normalMap;
		} background;
	} textures;
	
	// Vertex layout for the models
	vks::VertexLayout vertexLayout = vks::VertexLayout({
		vks::VERTEX_COMPONENT_POSITION,
		vks::VERTEX_COMPONENT_UV,
		vks::VERTEX_COMPONENT_COLOR,
		vks::VERTEX_COMPONENT_NORMAL,
		vks::VERTEX_COMPONENT_TANGENT,
	});

	struct {
		vks::Model model;
		vks::Model background;
		vks::Model quad;
	} models;

	struct {
		vk::PipelineVertexInputStateCreateInfo inputState;
		std::vector<vk::VertexInputBindingDescription> bindingDescriptions;
		std::vector<vk::VertexInputAttributeDescription> attributeDescriptions;
	} vertices;

	struct {
		glm::mat4 projection;
		glm::mat4 model;
		glm::mat4 view;
		glm::vec4 instancePos[3];
		int layer;
	} uboVS, uboOffscreenVS;

	// This UBO stores the shadow matrices for all of the light sources
	// The matrices are indexed using geometry shader instancing
	// The instancePos is used to place the models using instanced draws
	struct {
		glm::mat4 mvp[LIGHT_COUNT];
		glm::vec4 instancePos[3];
	} uboShadowGS;

	struct Light {
		glm::vec4 position;
		glm::vec4 target;
		glm::vec4 color;
		glm::mat4 viewMatrix;
	};

	struct {
		glm::vec4 viewPos;
		Light lights[LIGHT_COUNT];
		uint32_t useShadows = 1;
	} uboFragmentLights;

	struct {
		vks::Buffer vsFullScreen;
		vks::Buffer vsOffscreen;
		vks::Buffer fsLights;
		vks::Buffer uboShadowGS;
	} uniformBuffers;

	struct {
		vk::Pipeline deferred;
		vk::Pipeline offscreen;
		vk::Pipeline debug;
		vk::Pipeline shadowpass;
	} pipelines;

	struct {
		//todo: rename, shared with deferred and shadow pass
		vk::PipelineLayout deferred;
		vk::PipelineLayout offscreen;
	} pipelineLayouts;

	struct {
		vk::DescriptorSet model;
		vk::DescriptorSet background;
		vk::DescriptorSet shadow;
	} descriptorSets;

	vk::DescriptorSet descriptorSet;
	vk::DescriptorSetLayout descriptorSetLayout;

	struct
	{
		// Framebuffer resources for the deferred pass
		vks::Framebuffer *deferred;
		// Framebuffer resources for the shadow pass
		vks::Framebuffer *shadow;
	} frameBuffers;

	struct {
		vk::CommandBuffer deferred;
	} commandBuffers;

	// Semaphore used to synchronize between offscreen and final scene rendering
	vk::Semaphore offscreenSemaphore;

	VulkanExample() : VulkanExampleBase(ENABLE_VALIDATION)
	{
		enableTextOverlay = true;
		title = "Vulkan Example - Deferred shading with shadows (2016 by Sascha Willems)";
		camera.type = Camera::CameraType::firstperson;
#if defined(__ANDROID__)
		camera.movementSpeed = 2.5f;
#else
		camera.movementSpeed = 5.0f;
		camera.rotationSpeed = 0.25f;
#endif
		camera.position = { 2.15f, 0.3f, -8.75f };
		camera.setRotation(glm::vec3(-0.75f, 12.5f, 0.0f));
		camera.setPerspective(60.0f, (float)width / (float)height, zNear, zFar);
		timerSpeed *= 0.25f;
		paused = true;
	}

	~VulkanExample()
	{
		// Frame buffers
		if (frameBuffers.deferred)
		{
			delete frameBuffers.deferred;
		}
		if (frameBuffers.shadow)
		{
			delete frameBuffers.shadow;
		}

		device.destroyPipeline(pipelines.deferred);
		device.destroyPipeline(pipelines.offscreen);
		device.destroyPipeline(pipelines.shadowpass);
		device.destroyPipeline(pipelines.debug);

		device.destroyPipelineLayout(pipelineLayouts.deferred);
		device.destroyPipelineLayout(pipelineLayouts.offscreen);

		device.destroyDescriptorSetLayout(descriptorSetLayout);

		// Meshes
		models.model.destroy();
		models.background.destroy();
		models.quad.destroy();

		// Uniform buffers
		uniformBuffers.vsOffscreen.destroy();
		uniformBuffers.vsFullScreen.destroy();
		uniformBuffers.fsLights.destroy();
		uniformBuffers.uboShadowGS.destroy();

		device.freeCommandBuffers(cmdPool, commandBuffers.deferred);

		// Textures
		textures.model.colorMap.destroy();
		textures.model.normalMap.destroy();
		textures.background.colorMap.destroy();
		textures.background.normalMap.destroy();

		device.destroySemaphore(offscreenSemaphore);
	}

	// Enable physical device features required for this example				
	virtual void getEnabledFeatures()
	{
		// Geometry shader support is required for writing to multiple shadow map layers in one single pass
		if (deviceFeatures.geometryShader) {
			enabledFeatures.geometryShader = VK_TRUE;
		}
		else {
			vks::tools::exitFatal("Selected GPU does not support geometry shaders!", "Feature not supported");
		}
		// Enable anisotropic filtering if supported
		if (deviceFeatures.samplerAnisotropy) {
			enabledFeatures.samplerAnisotropy = VK_TRUE;
		}
		// Enable texture compression  
		if (deviceFeatures.textureCompressionBC) {
			enabledFeatures.textureCompressionBC = VK_TRUE;
		}
		else if (deviceFeatures.textureCompressionASTC_LDR) {
			enabledFeatures.textureCompressionASTC_LDR = VK_TRUE;
		}
		else if (deviceFeatures.textureCompressionETC2) {
			enabledFeatures.textureCompressionETC2 = VK_TRUE;
		}
	}

	// Prepare a layered shadow map with each layer containing depth from a light's point of view
	// The shadow mapping pass uses geometry shader instancing to output the scene from the different
	// light sources' point of view to the layers of the depth attachment in one single pass 
	void shadowSetup()
	{
		frameBuffers.shadow = new vks::Framebuffer(vulkanDevice);

		frameBuffers.shadow->width = SHADOWMAP_DIM;
		frameBuffers.shadow->height = SHADOWMAP_DIM;

		// Create a layered depth attachment for rendering the depth maps from the lights' point of view
		// Each layer corresponds to one of the lights
		// The actual output to the separate layers is done in the geometry shader using shader instancing
		// We will pass the matrices of the lights to the GS that selects the layer by the current invocation
		vks::AttachmentCreateInfo attachmentInfo = {};
		attachmentInfo.format = SHADOWMAP_FORMAT;
		attachmentInfo.width = SHADOWMAP_DIM;
		attachmentInfo.height = SHADOWMAP_DIM;
		attachmentInfo.layerCount = LIGHT_COUNT;
		attachmentInfo.usage = vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eSampled;
		frameBuffers.shadow->addAttachment(attachmentInfo);

		// Create sampler to sample from to depth attachment 
		// Used to sample in the fragment shader for shadowed rendering
		VK_CHECK_RESULT(frameBuffers.shadow->createSampler(vk::Filter::eLinear, vk::Filter::eLinear, vk::SamplerAddressMode::eClampToEdge));

		// Create default renderpass for the framebuffer
		VK_CHECK_RESULT(frameBuffers.shadow->createRenderPass());
	}

	// Prepare the framebuffer for offscreen rendering with multiple attachments used as render targets inside the fragment shaders
	void deferredSetup()
	{
		frameBuffers.deferred = new vks::Framebuffer(vulkanDevice);

		frameBuffers.deferred->width = FB_DIM;
		frameBuffers.deferred->height = FB_DIM;

		// Four attachments (3 color, 1 depth)
		vks::AttachmentCreateInfo attachmentInfo = {};
		attachmentInfo.width = FB_DIM;
		attachmentInfo.height = FB_DIM;
		attachmentInfo.layerCount = 1;
		attachmentInfo.usage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled;

		// Color attachments
		// Attachment 0: (World space) Positions
		attachmentInfo.format = vk::Format::eR16G16B16A16Sfloat;
		frameBuffers.deferred->addAttachment(attachmentInfo);

		// Attachment 1: (World space) Normals
		attachmentInfo.format = vk::Format::eR16G16B16A16Sfloat;
		frameBuffers.deferred->addAttachment(attachmentInfo);

		// Attachment 2: Albedo (color)
		attachmentInfo.format = vk::Format::eR8G8B8A8Unorm;
		frameBuffers.deferred->addAttachment(attachmentInfo);

		// Depth attachment
		// Find a suitable depth format
		vk::Format attDepthFormat;
		vk::Bool32 validDepthFormat = vks::tools::getSupportedDepthFormat(physicalDevice, &attDepthFormat);
		assert(validDepthFormat);

		attachmentInfo.format = attDepthFormat;
		attachmentInfo.usage = vk::ImageUsageFlagBits::eDepthStencilAttachment;
		frameBuffers.deferred->addAttachment(attachmentInfo);

		// Create sampler to sample from the color attachments
		VK_CHECK_RESULT(frameBuffers.deferred->createSampler(vk::Filter::eNearest, vk::Filter::eNearest, vk::SamplerAddressMode::eClampToEdge));

		// Create default renderpass for the framebuffer
		VK_CHECK_RESULT(frameBuffers.deferred->createRenderPass());
	}

	// Put render commands for the scene into the given command buffer
	void renderScene(vk::CommandBuffer cmdBuffer, bool shadow)
	{
		std::vector<vk::DeviceSize> offsets = { 0 };

		// Background
		cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayouts.offscreen, 0, shadow ? descriptorSets.shadow : descriptorSets.background, nullptr);
		cmdBuffer.bindVertexBuffers(VERTEX_BUFFER_BIND_ID, models.background.vertices.buffer, offsets);
		cmdBuffer.bindIndexBuffer(models.background.indices.buffer, 0, vk::IndexType::eUint32);
		cmdBuffer.drawIndexed(models.background.indexCount, 1, 0, 0, 0);

		// Objects
		cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayouts.offscreen, 0, shadow ? descriptorSets.shadow : descriptorSets.model, nullptr);
		cmdBuffer.bindVertexBuffers(VERTEX_BUFFER_BIND_ID, models.model.vertices.buffer, offsets);
		cmdBuffer.bindIndexBuffer(models.model.indices.buffer, 0, vk::IndexType::eUint32);
		cmdBuffer.drawIndexed(models.model.indexCount, 3, 0, 0, 0);
	}

	// Build a secondary command buffer for rendering the scene values to the offscreen frame buffer attachments
	void buildDeferredCommandBuffer()
	{
		if (!commandBuffers.deferred)
		{
			commandBuffers.deferred = VulkanExampleBase::createCommandBuffer(vk::CommandBufferLevel::ePrimary, false);
		}

		// Create a semaphore used to synchronize offscreen rendering and usage
		vk::SemaphoreCreateInfo semaphoreCreateInfo = vks::initializers::semaphoreCreateInfo();
		offscreenSemaphore = device.createSemaphore(semaphoreCreateInfo);

		vk::CommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();

		vk::RenderPassBeginInfo renderPassBeginInfo = vks::initializers::renderPassBeginInfo();
		std::array<vk::ClearValue, 4> clearValues = {};
		vk::Viewport viewport;
		vk::Rect2D scissor;

		// First pass: Shadow map generation
		// -------------------------------------------------------------------------------------------------------
	
		clearValues[0].depthStencil = vk::ClearDepthStencilValue{ 1.0f, 0 };

		renderPassBeginInfo.renderPass = frameBuffers.shadow->renderPass;
		renderPassBeginInfo.framebuffer = frameBuffers.shadow->framebuffer;
		renderPassBeginInfo.renderArea.extent.width = frameBuffers.shadow->width;
		renderPassBeginInfo.renderArea.extent.height = frameBuffers.shadow->height;
		renderPassBeginInfo.clearValueCount = 1;
		renderPassBeginInfo.pClearValues = clearValues.data();

		commandBuffers.deferred.begin(cmdBufInfo);

		viewport = vks::initializers::viewport((float)frameBuffers.shadow->width, (float)frameBuffers.shadow->height, 0.0f, 1.0f);
		commandBuffers.deferred.setViewport(0, viewport);

		scissor = vks::initializers::rect2D(frameBuffers.shadow->width, frameBuffers.shadow->height, 0, 0);
		commandBuffers.deferred.setScissor(0, scissor);

		// Set depth bias (aka "Polygon offset")
		vkCmdSetDepthBias(
			commandBuffers.deferred,
			depthBiasConstant,
			0.0f,
			depthBiasSlope);

		commandBuffers.deferred.beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);
		commandBuffers.deferred.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.shadowpass);
		renderScene(commandBuffers.deferred, true);
		commandBuffers.deferred.endRenderPass();

		// Second pass: Deferred calculations
		// -------------------------------------------------------------------------------------------------------

		// Clear values for all attachments written in the fragment sahder
		clearValues[0].color = vk::ClearColorValue{ std::array<float, 4>{ 0.0f, 0.0f, 0.0f, 0.0f } };
		clearValues[1].color = vk::ClearColorValue{ std::array<float, 4>{ 0.0f, 0.0f, 0.0f, 0.0f } };
		clearValues[2].color = vk::ClearColorValue{ std::array<float, 4>{ 0.0f, 0.0f, 0.0f, 0.0f } };
		clearValues[3].depthStencil = vk::ClearDepthStencilValue{ 1.0f, 0 };

		renderPassBeginInfo.renderPass = frameBuffers.deferred->renderPass;
		renderPassBeginInfo.framebuffer = frameBuffers.deferred->framebuffer;
		renderPassBeginInfo.renderArea.extent.width = frameBuffers.deferred->width;
		renderPassBeginInfo.renderArea.extent.height = frameBuffers.deferred->height;
		renderPassBeginInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
		renderPassBeginInfo.pClearValues = clearValues.data();

		commandBuffers.deferred.beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);

		viewport = vks::initializers::viewport((float)frameBuffers.deferred->width, (float)frameBuffers.deferred->height, 0.0f, 1.0f);
		commandBuffers.deferred.setViewport(0, viewport);

		scissor = vks::initializers::rect2D(frameBuffers.deferred->width, frameBuffers.deferred->height, 0, 0);
		commandBuffers.deferred.setScissor(0, scissor);

		commandBuffers.deferred.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.offscreen);
		renderScene(commandBuffers.deferred, false);
		commandBuffers.deferred.endRenderPass();

		commandBuffers.deferred.end();
	}

	void loadAssets()
	{
		models.model.loadFromFile(getAssetPath() + "models/armor/armor.dae", vertexLayout, 1.0f, vulkanDevice, queue);

		vks::ModelCreateInfo modelCreateInfo;
		modelCreateInfo.scale = glm::vec3(15.0f);
		modelCreateInfo.uvscale = glm::vec2(1.0f, 1.5f);
		modelCreateInfo.center = glm::vec3(0.0f, 2.3f, 0.0f);
		models.background.loadFromFile(getAssetPath() + "models/openbox.dae", vertexLayout, &modelCreateInfo, vulkanDevice, queue);

		// Textures
		std::string texFormatSuffix;
		vk::Format texFormat;
		// Get supported compressed texture format
		if (vulkanDevice->features.textureCompressionBC) {
			texFormatSuffix = "_bc3_unorm";
			texFormat = vk::Format::eBc3UnormBlock;
		}
		else if (vulkanDevice->features.textureCompressionASTC_LDR) {
			texFormatSuffix = "_astc_8x8_unorm";
			texFormat = vk::Format::eAstc8x8UnormBlock;
		}
		else if (vulkanDevice->features.textureCompressionETC2) {
			texFormatSuffix = "_etc2_unorm";
			texFormat = vk::Format::eEtc2R8G8B8A8UnormBlock;
		}
		else {
			vks::tools::exitFatal("Device does not support any compressed texture format!", "Error");
		}

		textures.model.colorMap.loadFromFile(getAssetPath() + "models/armor/color" + texFormatSuffix + ".ktx", texFormat, vulkanDevice, queue);
		textures.model.normalMap.loadFromFile(getAssetPath() + "models/armor/normal" + texFormatSuffix + ".ktx", texFormat, vulkanDevice, queue);
		textures.background.colorMap.loadFromFile(getAssetPath() + "textures/stonefloor02_color" + texFormatSuffix + ".ktx", texFormat, vulkanDevice, queue);
		textures.background.normalMap.loadFromFile(getAssetPath() + "textures/stonefloor02_normal" + texFormatSuffix + ".ktx", texFormat, vulkanDevice, queue);
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
		clearValues[0].color = vk::ClearColorValue{ std::array<float, 4>{ 0.0f, 0.0f, 0.2f, 0.0f } };
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
			renderPassBeginInfo.framebuffer = VulkanExampleBase::frameBuffers[i];

			drawCmdBuffers[i].begin(cmdBufInfo);

			drawCmdBuffers[i].beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);

			vk::Viewport viewport = vks::initializers::viewport((float)width, (float)height, 0.0f, 1.0f);
			drawCmdBuffers[i].setViewport(0, viewport);

			vk::Rect2D scissor = vks::initializers::rect2D(width, height, 0, 0);
			drawCmdBuffers[i].setScissor(0, scissor);

			std::vector<vk::DeviceSize> offsets = { 0 };
			drawCmdBuffers[i].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayouts.deferred, 0, descriptorSet, nullptr);

			// Final composition as full screen quad
			drawCmdBuffers[i].bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.deferred);
			drawCmdBuffers[i].bindVertexBuffers(VERTEX_BUFFER_BIND_ID, models.quad.vertices.buffer, offsets);
			drawCmdBuffers[i].bindIndexBuffer(models.quad.indices.buffer, 0, vk::IndexType::eUint32);
			drawCmdBuffers[i].drawIndexed(6, 1, 0, 0, 0);

			if (debugDisplay)
			{
				// Visualize depth maps
				drawCmdBuffers[i].bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.debug);
				drawCmdBuffers[i].drawIndexed(6, LIGHT_COUNT, 0, 0, 0);
			}

			drawCmdBuffers[i].endRenderPass();

			drawCmdBuffers[i].end();
		}
	}

	/** @brief Create a single quad for fullscreen deferred pass and debug passes (debug pass uses instancing for light visualization) */
	void generateQuads()
	{
		struct Vertex {
			float pos[3];
			float uv[2];
			float col[3];
			float normal[3];
			float tangent[3];
		};

		std::vector<Vertex> vertexBuffer;

		vertexBuffer.push_back(Vertex{ { 1.0f, 1.0f, 0.0f },{ 1.0f, 1.0f },{ 1.0f, 1.0f, 1.0f },{ 0.0f, 0.0f, 0.0f } });
		vertexBuffer.push_back(Vertex{ { 0.0f, 1.0f, 0.0f },{ 0.0f, 1.0f },{ 1.0f, 1.0f, 1.0f },{ 0.0f, 0.0f, 0.0f } });
		vertexBuffer.push_back(Vertex{ { 0.0f, 0.0f, 0.0f },{ 0.0f, 0.0f },{ 1.0f, 1.0f, 1.0f },{ 0.0f, 0.0f, 0.0f } });
		vertexBuffer.push_back(Vertex{ { 1.0f, 0.0f, 0.0f },{ 1.0f, 0.0f },{ 1.0f, 1.0f, 1.0f },{ 0.0f, 0.0f, 0.0f } });

		vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eVertexBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			vertexBuffer.size() * sizeof(Vertex),
			&models.quad.vertices.buffer,
			&models.quad.vertices.memory,
			vertexBuffer.data());

		// Setup indices
		std::vector<uint32_t> indexBuffer = { 0,1,2, 2,3,0 };
		for (uint32_t i = 0; i < 3; ++i)
		{
			uint32_t indices[6] = { 0,1,2, 2,3,0 };
			for (auto index : indices)
			{
				indexBuffer.push_back(i * 4 + index);
			}
		}
		models.quad.indexCount = static_cast<uint32_t>(indexBuffer.size());

		vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eIndexBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			indexBuffer.size() * sizeof(uint32_t),
			&models.quad.indices.buffer,
			&models.quad.indices.memory,
			indexBuffer.data());

		models.quad.device = device;
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
		vertices.attributeDescriptions.resize(5);
		// Location 0: Position
		vertices.attributeDescriptions[0] =
			vks::initializers::vertexInputAttributeDescription(
				VERTEX_BUFFER_BIND_ID,
				0,
				vk::Format::eR32G32B32Sfloat,
				0);
		// Location 1: Texture coordinates
		vertices.attributeDescriptions[1] =
			vks::initializers::vertexInputAttributeDescription(
				VERTEX_BUFFER_BIND_ID,
				1,
				vk::Format::eR32G32Sfloat,
				sizeof(float) * 3);
		// Location 2: Color
		vertices.attributeDescriptions[2] =
			vks::initializers::vertexInputAttributeDescription(
				VERTEX_BUFFER_BIND_ID,
				2,
				vk::Format::eR32G32B32Sfloat,
				sizeof(float) * 5);
		// Location 3: Normal
		vertices.attributeDescriptions[3] =
			vks::initializers::vertexInputAttributeDescription(
				VERTEX_BUFFER_BIND_ID,
				3,
				vk::Format::eR32G32B32Sfloat,
				sizeof(float) * 8);
		// Location 4: Tangent
		vertices.attributeDescriptions[4] =
			vks::initializers::vertexInputAttributeDescription(
				VERTEX_BUFFER_BIND_ID,
				4,
				vk::Format::eR32G32B32Sfloat,
				sizeof(float) * 11);

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
			vks::initializers::descriptorPoolSize(vk::DescriptorType::eUniformBuffer, 12), //todo: separate set layouts
			vks::initializers::descriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, 16)
		};

		vk::DescriptorPoolCreateInfo descriptorPoolInfo =
			vks::initializers::descriptorPoolCreateInfo(
				static_cast<uint32_t>(poolSizes.size()),
				poolSizes.data(),
				4);

		descriptorPool = device.createDescriptorPool(descriptorPoolInfo);
	}

	void setupDescriptorSetLayout()
	{
		// todo: split for clarity, esp. with GS instancing
		// Deferred shading layout (Shared with debug display)
		std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings =
		{
			// Binding 0: Vertex shader uniform buffer
			vks::initializers::descriptorSetLayoutBinding(
				vk::DescriptorType::eUniformBuffer,
				vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eGeometry,
				0),
			// Binding 1: Position texture
			vks::initializers::descriptorSetLayoutBinding(
				vk::DescriptorType::eCombinedImageSampler,
				vk::ShaderStageFlagBits::eFragment,
				1),
			// Binding 2: Normals texture
			vks::initializers::descriptorSetLayoutBinding(
				vk::DescriptorType::eCombinedImageSampler,
				vk::ShaderStageFlagBits::eFragment,
				2),
			// Binding 3: Albedo texture
			vks::initializers::descriptorSetLayoutBinding(
				vk::DescriptorType::eCombinedImageSampler,
				vk::ShaderStageFlagBits::eFragment,
				3),
			// Binding 4: Fragment shader uniform buffer
			vks::initializers::descriptorSetLayoutBinding(
				vk::DescriptorType::eUniformBuffer,
				vk::ShaderStageFlagBits::eFragment,
				4),
			// Binding 5: Shadow map
			vks::initializers::descriptorSetLayoutBinding(
				vk::DescriptorType::eCombinedImageSampler,
				vk::ShaderStageFlagBits::eFragment,
				5),
		};

		vk::DescriptorSetLayoutCreateInfo descriptorLayout =
			vks::initializers::descriptorSetLayoutCreateInfo(
				setLayoutBindings);

		descriptorSetLayout = device.createDescriptorSetLayout(descriptorLayout);

		vk::PipelineLayoutCreateInfo pPipelineLayoutCreateInfo =
			vks::initializers::pipelineLayoutCreateInfo(
				&descriptorSetLayout,
				1);

		pipelineLayouts.deferred = device.createPipelineLayout(pPipelineLayoutCreateInfo);

		// Offscreen (scene) rendering pipeline layout
		pipelineLayouts.offscreen = device.createPipelineLayout(pPipelineLayoutCreateInfo);
	}

	void setupDescriptorSet()
	{
		std::vector<vk::WriteDescriptorSet> writeDescriptorSets;

		// Textured quad descriptor set
		vk::DescriptorSetAllocateInfo allocInfo =
			vks::initializers::descriptorSetAllocateInfo(
				descriptorPool,
				&descriptorSetLayout,
				1);

		descriptorSet = device.allocateDescriptorSets(allocInfo)[0];

		// Image descriptors for the offscreen color attachments
		vk::DescriptorImageInfo texDescriptorPosition =
			vks::initializers::descriptorImageInfo(
				frameBuffers.deferred->sampler,
				frameBuffers.deferred->attachments[0].view,
				vk::ImageLayout::eShaderReadOnlyOptimal);

		vk::DescriptorImageInfo texDescriptorNormal =
			vks::initializers::descriptorImageInfo(
				frameBuffers.deferred->sampler,
				frameBuffers.deferred->attachments[1].view,
				vk::ImageLayout::eShaderReadOnlyOptimal);

		vk::DescriptorImageInfo texDescriptorAlbedo =
			vks::initializers::descriptorImageInfo(
				frameBuffers.deferred->sampler,
				frameBuffers.deferred->attachments[2].view,
				vk::ImageLayout::eShaderReadOnlyOptimal);

		vk::DescriptorImageInfo texDescriptorShadowMap =
			vks::initializers::descriptorImageInfo(
				frameBuffers.shadow->sampler,
				frameBuffers.shadow->attachments[0].view,
				vk::ImageLayout::eDepthStencilReadOnlyOptimal);

		writeDescriptorSets = {
			// Binding 0: Vertex shader uniform buffer
			vks::initializers::writeDescriptorSet(
				descriptorSet,
				vk::DescriptorType::eUniformBuffer,
				0,
				&uniformBuffers.vsFullScreen.descriptor),
			// Binding 1: World space position texture
			vks::initializers::writeDescriptorSet(
				descriptorSet,
				vk::DescriptorType::eCombinedImageSampler,
				1,
				&texDescriptorPosition),
			// Binding 2: World space normals texture
			vks::initializers::writeDescriptorSet(
				descriptorSet,
				vk::DescriptorType::eCombinedImageSampler,
				2,
				&texDescriptorNormal),
			// Binding 3: Albedo texture
			vks::initializers::writeDescriptorSet(
				descriptorSet,
				vk::DescriptorType::eCombinedImageSampler,
				3,
				&texDescriptorAlbedo),
			// Binding 4: Fragment shader uniform buffer
			vks::initializers::writeDescriptorSet(
				descriptorSet,
				vk::DescriptorType::eUniformBuffer,
				4,
				&uniformBuffers.fsLights.descriptor),
			// Binding 5: Shadow map
			vks::initializers::writeDescriptorSet(
				descriptorSet,
				vk::DescriptorType::eCombinedImageSampler,
				5,
				&texDescriptorShadowMap),
		};

		device.updateDescriptorSets(writeDescriptorSets, nullptr);

		// Offscreen (scene)

		// Model
		descriptorSets.model = device.allocateDescriptorSets(allocInfo)[0];
		writeDescriptorSets =
		{
			// Binding 0: Vertex shader uniform buffer
			vks::initializers::writeDescriptorSet(
			descriptorSets.model,
				vk::DescriptorType::eUniformBuffer,
				0,
				&uniformBuffers.vsOffscreen.descriptor),
			// Binding 1: Color map
			vks::initializers::writeDescriptorSet(
				descriptorSets.model,
				vk::DescriptorType::eCombinedImageSampler,
				1,
				&textures.model.colorMap.descriptor),
			// Binding 2: Normal map
			vks::initializers::writeDescriptorSet(
				descriptorSets.model,
				vk::DescriptorType::eCombinedImageSampler,
				2,
				&textures.model.normalMap.descriptor)
		};
		device.updateDescriptorSets(writeDescriptorSets, nullptr);

		// Background
		descriptorSets.background = device.allocateDescriptorSets(allocInfo)[0];
		writeDescriptorSets =
		{
			// Binding 0: Vertex shader uniform buffer
			vks::initializers::writeDescriptorSet(
				descriptorSets.background,
				vk::DescriptorType::eUniformBuffer,
				0,
				&uniformBuffers.vsOffscreen.descriptor),
			// Binding 1: Color map
			vks::initializers::writeDescriptorSet(
				descriptorSets.background,
				vk::DescriptorType::eCombinedImageSampler,
				1,
				&textures.background.colorMap.descriptor),
			// Binding 2: Normal map
			vks::initializers::writeDescriptorSet(
				descriptorSets.background,
				vk::DescriptorType::eCombinedImageSampler,
				2,
				&textures.background.normalMap.descriptor)
		};
		device.updateDescriptorSets(writeDescriptorSets, nullptr);

		// Shadow mapping
		descriptorSets.shadow = device.allocateDescriptorSets(allocInfo)[0];
		writeDescriptorSets =
		{
			// Binding 0: Vertex shader uniform buffer
			vks::initializers::writeDescriptorSet(
				descriptorSets.shadow,
				vk::DescriptorType::eUniformBuffer,
				0,
				&uniformBuffers.uboShadowGS.descriptor),
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

		// Final fullscreen pass pipeline
		std::array<vk::PipelineShaderStageCreateInfo, 2> shaderStages;

		shaderStages[0] = loadShader(getAssetPath() + "shaders/deferredshadows/deferred.vert.spv", vk::ShaderStageFlagBits::eVertex);
		shaderStages[1] = loadShader(getAssetPath() + "shaders/deferredshadows/deferred.frag.spv", vk::ShaderStageFlagBits::eFragment);

		vk::GraphicsPipelineCreateInfo pipelineCreateInfo =
			vks::initializers::pipelineCreateInfo(
				pipelineLayouts.deferred,
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

		pipelines.deferred = device.createGraphicsPipelines(pipelineCache, pipelineCreateInfo)[0];

		// Debug display pipeline
		shaderStages[0] = loadShader(getAssetPath() + "shaders/deferredshadows/debug.vert.spv", vk::ShaderStageFlagBits::eVertex);
		shaderStages[1] = loadShader(getAssetPath() + "shaders/deferredshadows/debug.frag.spv", vk::ShaderStageFlagBits::eFragment);
		pipelines.debug = device.createGraphicsPipelines(pipelineCache, pipelineCreateInfo)[0];

		// Offscreen pipeline
		shaderStages[0] = loadShader(getAssetPath() + "shaders/deferredshadows/mrt.vert.spv", vk::ShaderStageFlagBits::eVertex);
		shaderStages[1] = loadShader(getAssetPath() + "shaders/deferredshadows/mrt.frag.spv", vk::ShaderStageFlagBits::eFragment);

		// Separate render pass
		pipelineCreateInfo.renderPass = frameBuffers.deferred->renderPass;

		// Separate layout
		pipelineCreateInfo.layout = pipelineLayouts.offscreen;

		// Blend attachment states required for all color attachments
		// This is important, as color write mask will otherwise be 0x0 and you
		// won't see anything rendered to the attachment
		vk::PipelineColorBlendAttachmentState tmp = vks::initializers::pipelineColorBlendAttachmentState(
			vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA, 
			VK_FALSE);
		std::array<vk::PipelineColorBlendAttachmentState, 3> blendAttachmentStates = { tmp, tmp, tmp };

		colorBlendState.attachmentCount = static_cast<uint32_t>(blendAttachmentStates.size());
		colorBlendState.pAttachments = blendAttachmentStates.data();

		pipelines.offscreen = device.createGraphicsPipelines(pipelineCache, pipelineCreateInfo)[0];

		// Shadow mapping pipeline
		// The shadow mapping pipeline uses geometry shader instancing (invocations layout modifier) to output 
		// shadow maps for multiple lights sources into the different shadow map layers in one single render pass
		std::array<vk::PipelineShaderStageCreateInfo, 3> shadowStages;
		shadowStages[0] = loadShader(getAssetPath() + "shaders/deferredshadows/shadow.vert.spv", vk::ShaderStageFlagBits::eVertex);
		shadowStages[1] = loadShader(getAssetPath() + "shaders/deferredshadows/shadow.frag.spv", vk::ShaderStageFlagBits::eFragment);
		shadowStages[2] = loadShader(getAssetPath() + "shaders/deferredshadows/shadow.geom.spv", vk::ShaderStageFlagBits::eGeometry);

		pipelineCreateInfo.pStages = shadowStages.data();
		pipelineCreateInfo.stageCount = static_cast<uint32_t>(shadowStages.size());
		
		// Shadow pass doesn't use any color attachments
		colorBlendState.attachmentCount = 0;
		colorBlendState.pAttachments = nullptr;
		// Cull front faces
		rasterizationState.cullMode = vk::CullModeFlagBits::eFront;
		depthStencilState.depthCompareOp = vk::CompareOp::eLessOrEqual;
		// Enable depth bias
		rasterizationState.depthBiasEnable = VK_TRUE;
		// Add depth bias to dynamic state, so we can change it at runtime
		dynamicStateEnables.push_back(vk::DynamicState::eDepthBias);
		dynamicState =
			vks::initializers::pipelineDynamicStateCreateInfo(
				dynamicStateEnables);
		// Reset blend attachment state
		pipelineCreateInfo.renderPass = frameBuffers.shadow->renderPass;
		pipelines.shadowpass = device.createGraphicsPipelines(pipelineCache, pipelineCreateInfo)[0];
	}

	// Prepare and initialize uniform buffer containing shader uniforms
	void prepareUniformBuffers()
	{
		// Fullscreen vertex shader
		vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eUniformBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			&uniformBuffers.vsFullScreen,
			sizeof(uboVS));

		// Deferred vertex shader
		vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eUniformBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			&uniformBuffers.vsOffscreen,
			sizeof(uboOffscreenVS));

		// Deferred fragment shader
		vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eUniformBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			&uniformBuffers.fsLights,
			sizeof(uboFragmentLights));

		// Shadow map vertex shader (matrices from shadow's pov)
		vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eUniformBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			&uniformBuffers.uboShadowGS,
			sizeof(uboShadowGS));

		// Map persistent
		uniformBuffers.vsFullScreen.map();
		uniformBuffers.vsOffscreen.map();
		uniformBuffers.fsLights.map();
		uniformBuffers.uboShadowGS.map();

		// Init some values
		uboOffscreenVS.instancePos[0] = glm::vec4(0.0f);
		uboOffscreenVS.instancePos[1] = glm::vec4(-4.0f, 0.0, -4.0f, 0.0f);
		uboOffscreenVS.instancePos[2] = glm::vec4(4.0f, 0.0, -4.0f, 0.0f);

		uboOffscreenVS.instancePos[1] = glm::vec4(-7.0f, 0.0, -4.0f, 0.0f);
		uboOffscreenVS.instancePos[2] = glm::vec4(4.0f, 0.0, -6.0f, 0.0f);

		// Update
		updateUniformBuffersScreen();
		updateUniformBufferDeferredMatrices();
		updateUniformBufferDeferredLights();
	}

	void updateUniformBuffersScreen()
	{
		uboVS.projection = glm::ortho(0.0f, 1.0f, 0.0f, 1.0f, -1.0f, 1.0f);
		uboVS.model = glm::mat4();
		memcpy(uniformBuffers.vsFullScreen.mapped, &uboVS, sizeof(uboVS));
	}

	void updateUniformBufferDeferredMatrices()
	{
		uboOffscreenVS.projection = camera.matrices.perspective;
		uboOffscreenVS.view = camera.matrices.view;
		uboOffscreenVS.model = glm::mat4();
		memcpy(uniformBuffers.vsOffscreen.mapped, &uboOffscreenVS, sizeof(uboOffscreenVS));
	}

	Light initLight(glm::vec3 pos, glm::vec3 target, glm::vec3 color)
	{
		Light light;
		light.position = glm::vec4(pos, 1.0f);
		light.target = glm::vec4(target, 0.0f);
		light.color = glm::vec4(color, 0.0f);
		return light;
	}

	void initLights()
	{
		uboFragmentLights.lights[0] = initLight(glm::vec3(-14.0f, -0.5f, 15.0f), glm::vec3(-2.0f, 0.0f, 0.0f), glm::vec3(1.0f, 0.5f, 0.5f));
		uboFragmentLights.lights[1] = initLight(glm::vec3(14.0f, -4.0f, 12.0f), glm::vec3(2.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
		uboFragmentLights.lights[2] = initLight(glm::vec3(0.0f, -10.0f, 4.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f));
	}

	// Update fragment shader light position uniform block
	void updateUniformBufferDeferredLights()
	{
		// Animate
		//if (!paused)
		{
			uboFragmentLights.lights[0].position.x = -14.0f + std::abs(sin(glm::radians(timer * 360.0f)) * 20.0f);
			uboFragmentLights.lights[0].position.z = 15.0f + cos(glm::radians(timer *360.0f)) * 1.0f;

			uboFragmentLights.lights[1].position.x = 14.0f - std::abs(sin(glm::radians(timer * 360.0f)) * 2.5f);
			uboFragmentLights.lights[1].position.z = 13.0f + cos(glm::radians(timer *360.0f)) * 4.0f;

			uboFragmentLights.lights[2].position.x = 0.0f + sin(glm::radians(timer *360.0f)) * 4.0f;
			uboFragmentLights.lights[2].position.z = 4.0f + cos(glm::radians(timer *360.0f)) * 2.0f;
		}

		for (uint32_t i = 0; i < LIGHT_COUNT; i++)
		{
			// mvp from light's pov (for shadows)
			glm::mat4 shadowProj = glm::perspective(glm::radians(lightFOV), 1.0f, zNear, zFar);
			glm::mat4 shadowView = glm::lookAt(glm::vec3(uboFragmentLights.lights[i].position), glm::vec3(uboFragmentLights.lights[i].target), glm::vec3(0.0f, 1.0f, 0.0f));
			glm::mat4 shadowModel = glm::mat4();

			uboShadowGS.mvp[i] = shadowProj * shadowView * shadowModel;
			uboFragmentLights.lights[i].viewMatrix = uboShadowGS.mvp[i];
		}

		memcpy(uboShadowGS.instancePos, uboOffscreenVS.instancePos, sizeof(uboOffscreenVS.instancePos));

		memcpy(uniformBuffers.uboShadowGS.mapped, &uboShadowGS, sizeof(uboShadowGS));

		uboFragmentLights.viewPos = glm::vec4(camera.position, 0.0f) * glm::vec4(-1.0f, 1.0f, -1.0f, 1.0f);;
	
		memcpy(uniformBuffers.fsLights.mapped, &uboFragmentLights, sizeof(uboFragmentLights));
	}

	void draw()
	{
		VulkanExampleBase::prepareFrame();

		// Offscreen rendering

		// Wait for swap chain presentation to finish
		submitInfo.pWaitSemaphores = &semaphores.presentComplete;
		// Signal ready with offscreen semaphore
		submitInfo.pSignalSemaphores = &offscreenSemaphore;

		// Submit work

		// Shadow map pass
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &commandBuffers.deferred;
		queue.submit(submitInfo, vk::Fence(nullptr));

		// Scene rendering

		// Wait for offscreen semaphore
		submitInfo.pWaitSemaphores = &offscreenSemaphore;
		// Signal ready with render complete semaphpre
		submitInfo.pSignalSemaphores = &semaphores.renderComplete;

		// Submit work
		submitInfo.pCommandBuffers = &drawCmdBuffers[currentBuffer];
		queue.submit(submitInfo, vk::Fence(nullptr));

		VulkanExampleBase::submitFrame();
	}

	void prepare()
	{
		VulkanExampleBase::prepare();
		loadAssets();
		generateQuads();
		setupVertexDescriptions();
		deferredSetup();
		shadowSetup();
		initLights();
		prepareUniformBuffers();
		setupDescriptorSetLayout();
		preparePipelines();
		setupDescriptorPool();
		setupDescriptorSet();
		buildCommandBuffers();
		buildDeferredCommandBuffer();
		prepared = true;
	}

	virtual void render()
	{
		if (!prepared)
			return;
		draw();
		updateUniformBufferDeferredLights();
	}

	virtual void viewChanged()
	{
		updateUniformBufferDeferredMatrices();
	}

	void toggleDebugDisplay()
	{
		debugDisplay = !debugDisplay;
		reBuildCommandBuffers();
		updateUniformBuffersScreen();
	}

	void toggleShadows()
	{
		uboFragmentLights.useShadows = !uboFragmentLights.useShadows;
		updateUniformBufferDeferredLights();
	}

	virtual void keyPressed(uint32_t keyCode)
	{
		switch (keyCode)
		{
		case KEY_F1:
		case GAMEPAD_BUTTON_A:
			toggleDebugDisplay();
			updateTextOverlay();
			break;
		case KEY_F2:
		case GAMEPAD_BUTTON_X:
			toggleShadows();
			updateTextOverlay();
			break;
		}
	}

	virtual void getOverlayText(VulkanTextOverlay *textOverlay)
	{
#if defined(__ANDROID__)
		textOverlay->addText("Press \"Button A\" to toggle debug view", 5.0f, 85.0f, VulkanTextOverlay::alignLeft);
		textOverlay->addText("Press \"Button X\" to toggle shadows", 5.0f, 100.0f, VulkanTextOverlay::alignLeft);
#else
		textOverlay->addText("Press \"F1\" to toggle debug view", 5.0f, 85.0f, VulkanTextOverlay::alignLeft);
		textOverlay->addText("Press \"F2\" to toggle shadows", 5.0f, 100.0f, VulkanTextOverlay::alignLeft);
#endif
	}
};

VULKAN_EXAMPLE_MAIN()
