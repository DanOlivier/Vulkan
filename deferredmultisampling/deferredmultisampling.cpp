/*
* Vulkan Example - Multi sampling with explicit resolve for deferred shading example
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
#include "VulkanTexture.hpp"
#include "VulkanModel.hpp"

#define VERTEX_BUFFER_BIND_ID 0
#define ENABLE_VALIDATION false
// todo: check if hardware supports sample number (or select max. supported)
vk::SampleCountFlagBits SAMPLE_COUNT = vk::SampleCountFlagBits::e8;

class VulkanExample : public VulkanExampleBase
{
public:
	bool debugDisplay = false;
	bool useMSAA = true;
	bool useSampleShading = true;

	struct {
		struct {
			vks::Texture2D colorMap;
			vks::Texture2D normalMap;
		} model;
		struct {
			vks::Texture2D colorMap;
			vks::Texture2D normalMap;
		} floor;
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
		vks::Model floor;
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
	} uboVS, uboOffscreenVS;

	struct Light {
		glm::vec4 position;
		glm::vec3 color;
		float radius;
	};

	struct {
		Light lights[6];
		glm::vec4 viewPos;
		glm::ivec2 windowSize;
	} uboFragmentLights;

	struct {
		vks::Buffer vsFullScreen;
		vks::Buffer vsOffscreen;
		vks::Buffer fsLights;
	} uniformBuffers;

	struct {
		vk::Pipeline deferred;				// Deferred lighting calculation
		vk::Pipeline deferredNoMSAA;			// Deferred lighting calculation with explicit MSAA resolve
		vk::Pipeline offscreen;				// (Offscreen) scene rendering (fill G-Buffers)
		vk::Pipeline offscreenSampleShading;	// (Offscreen) scene rendering (fill G-Buffers) with sample shading rate enabled
		vk::Pipeline debug;					// G-Buffers debug display
	} pipelines;

	struct {
		vk::PipelineLayout deferred; 
		vk::PipelineLayout offscreen;
	} pipelineLayouts;

	struct {
		vk::DescriptorSet model;
		vk::DescriptorSet floor;
	} descriptorSets;

	vk::DescriptorSet descriptorSet;
	vk::DescriptorSetLayout descriptorSetLayout;

	// Framebuffer for offscreen rendering
	struct FrameBufferAttachment {
		vk::Image image;
		vk::DeviceMemory mem;
		vk::ImageView view;
		vk::Format format;
	};
	struct FrameBuffer {
		int32_t width, height;
		vk::Framebuffer frameBuffer;		
		FrameBufferAttachment position, normal, albedo;
		FrameBufferAttachment depth;
		vk::RenderPass renderPass;
	} offScreenFrameBuf;
	
	// One sampler for the frame buffer color attachments
	vk::Sampler colorSampler;

	vk::CommandBuffer offScreenCmdBuffer;

	// Semaphore used to synchronize between offscreen and final scene rendering
	vk::Semaphore offscreenSemaphore;

	VulkanExample() : VulkanExampleBase(ENABLE_VALIDATION)
	{
		zoom = -8.0f;
		rotation = { 0.0f, 0.0f, 0.0f };
		enableTextOverlay = true;
		title = "Vulkan Example - Deferred shading (2016 by Sascha Willems)";
		camera.type = Camera::CameraType::firstperson;
		camera.movementSpeed = 5.0f;
#ifndef __ANDROID__
		camera.rotationSpeed = 0.25f;
#endif
		camera.position = { 2.15f, 0.3f, -8.75f };
		camera.setRotation(glm::vec3(-0.75f, 12.5f, 0.0f));
		camera.setPerspective(60.0f, (float)width / (float)height, 0.1f, 256.0f);
		paused = true;
	}

	~VulkanExample()
	{
		// Clean up used Vulkan resources 
		// Note : Inherited destructor cleans up resources stored in base class

		device.destroySampler(colorSampler);

		// Frame buffer

		// Color attachments
		device.destroyImageView(offScreenFrameBuf.position.view);
		device.destroyImage(offScreenFrameBuf.position.image);
		device.freeMemory(offScreenFrameBuf.position.mem);

		device.destroyImageView(offScreenFrameBuf.normal.view);
		device.destroyImage(offScreenFrameBuf.normal.image);
		device.freeMemory(offScreenFrameBuf.normal.mem);

		device.destroyImageView(offScreenFrameBuf.albedo.view);
		device.destroyImage(offScreenFrameBuf.albedo.image);
		device.freeMemory(offScreenFrameBuf.albedo.mem);

		// Depth attachment
		device.destroyImageView(offScreenFrameBuf.depth.view);
		device.destroyImage(offScreenFrameBuf.depth.image);
		device.freeMemory(offScreenFrameBuf.depth.mem);

		device.destroyFramebuffer(offScreenFrameBuf.frameBuffer);

		device.destroyPipeline(pipelines.deferred);
		device.destroyPipeline(pipelines.deferredNoMSAA);
		device.destroyPipeline(pipelines.offscreen);
		device.destroyPipeline(pipelines.offscreenSampleShading);
		device.destroyPipeline(pipelines.debug);

		device.destroyPipelineLayout(pipelineLayouts.deferred);
		device.destroyPipelineLayout(pipelineLayouts.offscreen);

		device.destroyDescriptorSetLayout(descriptorSetLayout);

		// Meshes
		models.model.destroy();
		models.floor.destroy();

		// Uniform buffers
		uniformBuffers.vsOffscreen.destroy();
		uniformBuffers.vsFullScreen.destroy();
		uniformBuffers.fsLights.destroy();

		device.freeCommandBuffers(cmdPool, offScreenCmdBuffer);

		device.destroyRenderPass(offScreenFrameBuf.renderPass);

		textures.model.colorMap.destroy();
		textures.model.normalMap.destroy();
		textures.floor.colorMap.destroy();
		textures.floor.normalMap.destroy();

		device.destroySemaphore(offscreenSemaphore);
	}

	// Enable physical device features required for this example				
	virtual void getEnabledFeatures()
	{
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
	};

	// Create a frame buffer attachment
	void createAttachment(
		vk::Format format,  
		vk::ImageUsageFlagBits usage,
		FrameBufferAttachment *attachment,
		vk::CommandBuffer layoutCmd)
	{
		vk::ImageAspectFlags aspectMask;
		//vk::ImageLayout imageLayout;

		attachment->format = format;

		if (vk::ImageUsageFlags(usage) & vk::ImageUsageFlagBits::eColorAttachment)
		{
			aspectMask = vk::ImageAspectFlagBits::eColor;
			//imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
		}
		if (vk::ImageUsageFlags(usage) & vk::ImageUsageFlagBits::eDepthStencilAttachment)
		{
			aspectMask = vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil;
			//imageLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal;
		}

		//assert(aspectMask > 0);

		vk::ImageCreateInfo image = vks::initializers::imageCreateInfo();
		image.imageType = vk::ImageType::e2D;
		image.format = format;
		image.extent.width = offScreenFrameBuf.width;
		image.extent.height = offScreenFrameBuf.height;
		image.extent.depth = 1;
		image.mipLevels = 1;
		image.arrayLayers = 1;
		image.samples = SAMPLE_COUNT;
		image.tiling = vk::ImageTiling::eOptimal;
		image.usage = usage | vk::ImageUsageFlagBits::eSampled;

		vk::MemoryAllocateInfo memAlloc = vks::initializers::memoryAllocateInfo();
		vk::MemoryRequirements memReqs;

		attachment->image = device.createImage(image);
		memReqs = device.getImageMemoryRequirements(attachment->image);
		memAlloc.allocationSize = memReqs.size;
		memAlloc.memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);
		attachment->mem = device.allocateMemory(memAlloc);
		device.bindImageMemory(attachment->image, attachment->mem, 0);
		
		vk::ImageViewCreateInfo imageView = vks::initializers::imageViewCreateInfo();
		imageView.viewType = vk::ImageViewType::e2D;
		imageView.format = format;
		//imageView.subresourceRange = {};
		imageView.subresourceRange.aspectMask = aspectMask;
		imageView.subresourceRange.baseMipLevel = 0;
		imageView.subresourceRange.levelCount = 1;
		imageView.subresourceRange.baseArrayLayer = 0;
		imageView.subresourceRange.layerCount = 1;
		imageView.image = attachment->image;
		attachment->view = device.createImageView(imageView);
	}

	// Prepare a new framebuffer for offscreen rendering
	// The contents of this framebuffer are then
	// blitted to our render target
	void prepareOffscreenFramebuffer()
	{
		vk::CommandBuffer layoutCmd = VulkanExampleBase::createCommandBuffer(vk::CommandBufferLevel::ePrimary, true);

		offScreenFrameBuf.width = this->width;
		offScreenFrameBuf.height = this->height;

		//offScreenFrameBuf.width = FB_DIM;
		//offScreenFrameBuf.height = FB_DIM;

		// Color attachments

		// (World space) Positions
		createAttachment(
			vk::Format::eR16G16B16A16Sfloat,
			vk::ImageUsageFlagBits::eColorAttachment,
			&offScreenFrameBuf.position,
			layoutCmd);

		// (World space) Normals
		createAttachment(
			vk::Format::eR16G16B16A16Sfloat,
			vk::ImageUsageFlagBits::eColorAttachment,
			&offScreenFrameBuf.normal,
			layoutCmd);

		// Albedo (color)
		createAttachment(
			vk::Format::eR8G8B8A8Unorm,
			vk::ImageUsageFlagBits::eColorAttachment,
			&offScreenFrameBuf.albedo,
			layoutCmd);

		// Depth attachment

		// Find a suitable depth format
		vk::Format attDepthFormat;
		vk::Bool32 validDepthFormat = vks::tools::getSupportedDepthFormat(physicalDevice, &attDepthFormat);
		assert(validDepthFormat);

		createAttachment(
			attDepthFormat,
			vk::ImageUsageFlagBits::eDepthStencilAttachment,
			&offScreenFrameBuf.depth,
			layoutCmd);

		VulkanExampleBase::flushCommandBuffer(layoutCmd, queue, true);

		// Set up separate renderpass with references
		// to the color and depth attachments

		std::array<vk::AttachmentDescription, 4> attachmentDescs = {};

		// Init attachment properties
		for (uint32_t i = 0; i < 4; ++i)
		{
			attachmentDescs[i].samples = SAMPLE_COUNT;
			attachmentDescs[i].loadOp = vk::AttachmentLoadOp::eClear;
			attachmentDescs[i].storeOp = vk::AttachmentStoreOp::eStore;
			attachmentDescs[i].stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
			attachmentDescs[i].stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
			if (i == 3)
			{
				attachmentDescs[i].initialLayout = vk::ImageLayout::eUndefined;
				attachmentDescs[i].finalLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal;
			}
			else
			{
				attachmentDescs[i].initialLayout = vk::ImageLayout::eUndefined;
				attachmentDescs[i].finalLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
			}
		}

		// Formats
		attachmentDescs[0].format = offScreenFrameBuf.position.format;
		attachmentDescs[1].format = offScreenFrameBuf.normal.format;
		attachmentDescs[2].format = offScreenFrameBuf.albedo.format;
		attachmentDescs[3].format = offScreenFrameBuf.depth.format;

		std::vector<vk::AttachmentReference> colorReferences;
		colorReferences.push_back({ 0, vk::ImageLayout::eColorAttachmentOptimal });
		colorReferences.push_back({ 1, vk::ImageLayout::eColorAttachmentOptimal });
		colorReferences.push_back({ 2, vk::ImageLayout::eColorAttachmentOptimal });

		vk::AttachmentReference depthReference = {};
		depthReference.attachment = 3;
		depthReference.layout = vk::ImageLayout::eDepthStencilAttachmentOptimal;

		vk::SubpassDescription subpass = {};
		subpass.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
		subpass.pColorAttachments = colorReferences.data();
		subpass.colorAttachmentCount = static_cast<uint32_t>(colorReferences.size());
		subpass.pDepthStencilAttachment = &depthReference;

		// Use subpass dependencies for attachment layput transitions
		std::array<vk::SubpassDependency, 2> dependencies;

		dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
		dependencies[0].dstSubpass = 0;
		dependencies[0].srcStageMask = vk::PipelineStageFlagBits::eBottomOfPipe;
		dependencies[0].dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
		dependencies[0].srcAccessMask = vk::AccessFlagBits::eMemoryRead;
		dependencies[0].dstAccessMask = vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite;
		dependencies[0].dependencyFlags = vk::DependencyFlagBits::eByRegion;

		dependencies[1].srcSubpass = 0;													
		dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;								
		dependencies[1].srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
		dependencies[1].dstStageMask = vk::PipelineStageFlagBits::eBottomOfPipe;
		dependencies[1].srcAccessMask = vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite;
		dependencies[1].dstAccessMask = vk::AccessFlagBits::eMemoryRead;
		dependencies[1].dependencyFlags = vk::DependencyFlagBits::eByRegion;

		vk::RenderPassCreateInfo renderPassInfo = {};

		renderPassInfo.pAttachments = attachmentDescs.data();
		renderPassInfo.attachmentCount = static_cast<uint32_t>(attachmentDescs.size());
		renderPassInfo.subpassCount = 1;
		renderPassInfo.pSubpasses = &subpass;
		renderPassInfo.dependencyCount = 2;
		renderPassInfo.pDependencies = dependencies.data();
	
		offScreenFrameBuf.renderPass = device.createRenderPass(renderPassInfo);
	
		std::array<vk::ImageView,4> attachments;
		attachments[0] = offScreenFrameBuf.position.view;
		attachments[1] = offScreenFrameBuf.normal.view;
		attachments[2] = offScreenFrameBuf.albedo.view;
		attachments[3] = offScreenFrameBuf.depth.view;

		vk::FramebufferCreateInfo fbufCreateInfo = {};
		fbufCreateInfo.renderPass = offScreenFrameBuf.renderPass;
		fbufCreateInfo.pAttachments = attachments.data();
		fbufCreateInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
		fbufCreateInfo.width = offScreenFrameBuf.width;
		fbufCreateInfo.height = offScreenFrameBuf.height;
		fbufCreateInfo.layers = 1;
		offScreenFrameBuf.frameBuffer = device.createFramebuffer(fbufCreateInfo);

		// Create sampler to sample from the color attachments
		vk::SamplerCreateInfo sampler = vks::initializers::samplerCreateInfo();
		sampler.magFilter = vk::Filter::eNearest;
		sampler.minFilter = vk::Filter::eNearest;
		sampler.mipmapMode = vk::SamplerMipmapMode::eLinear;
		sampler.addressModeU = vk::SamplerAddressMode::eClampToEdge;
		sampler.addressModeV = sampler.addressModeU;
		sampler.addressModeW = sampler.addressModeU;
		sampler.mipLodBias = 0.0f;
		sampler.maxAnisotropy = 1.0f;
		sampler.minLod = 0.0f;
		sampler.maxLod = 1.0f;
		sampler.borderColor = vk::BorderColor::eFloatOpaqueWhite;
		colorSampler = device.createSampler(sampler);
	}

	// Build command buffer for rendering the scene to the offscreen frame buffer attachments
	void buildDeferredCommandBuffer()
	{
		if (!offScreenCmdBuffer)
		{
			offScreenCmdBuffer = VulkanExampleBase::createCommandBuffer(vk::CommandBufferLevel::ePrimary, false);
		}

		// Create a semaphore used to synchronize offscreen rendering and usage
		vk::SemaphoreCreateInfo semaphoreCreateInfo = vks::initializers::semaphoreCreateInfo();
		offscreenSemaphore = device.createSemaphore(semaphoreCreateInfo);

		vk::CommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();

		// Clear values for all attachments written in the fragment sahder
		std::array<vk::ClearValue,4> clearValues;
		clearValues[0].color = clearValues[1].color = vk::ClearColorValue{ std::array<float, 4>{ 0.0f, 0.0f, 0.0f, 0.0f } };
		clearValues[2].color = vk::ClearColorValue{ std::array<float, 4>{ 0.0f, 0.0f, 0.0f, 0.0f } };
		clearValues[3].depthStencil = vk::ClearDepthStencilValue{ 1.0f, 0 };

		vk::RenderPassBeginInfo renderPassBeginInfo = vks::initializers::renderPassBeginInfo();
		renderPassBeginInfo.renderPass =  offScreenFrameBuf.renderPass;
		renderPassBeginInfo.framebuffer = offScreenFrameBuf.frameBuffer;
		renderPassBeginInfo.renderArea.extent.width = offScreenFrameBuf.width;
		renderPassBeginInfo.renderArea.extent.height = offScreenFrameBuf.height;
		renderPassBeginInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
		renderPassBeginInfo.pClearValues = clearValues.data();

		offScreenCmdBuffer.begin(cmdBufInfo);

		offScreenCmdBuffer.beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);

		vk::Viewport viewport = vks::initializers::viewport((float)offScreenFrameBuf.width, (float)offScreenFrameBuf.height, 0.0f, 1.0f);
		offScreenCmdBuffer.setViewport(0, viewport);

		vk::Rect2D scissor = vks::initializers::rect2D(offScreenFrameBuf.width, offScreenFrameBuf.height, 0, 0);
		offScreenCmdBuffer.setScissor(0, scissor);

		offScreenCmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, useSampleShading ? pipelines.offscreenSampleShading : pipelines.offscreen);

		std::vector<vk::DeviceSize> offsets = { 0 };

		// Background
		offScreenCmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayouts.offscreen, 0, descriptorSets.floor, nullptr);
		offScreenCmdBuffer.bindVertexBuffers(VERTEX_BUFFER_BIND_ID, models.floor.vertices.buffer, offsets);
		offScreenCmdBuffer.bindIndexBuffer(models.floor.indices.buffer, 0, vk::IndexType::eUint32);
		offScreenCmdBuffer.drawIndexed(models.floor.indexCount, 1, 0, 0, 0);

		// Object
		offScreenCmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayouts.offscreen, 0, descriptorSets.model, nullptr);
		offScreenCmdBuffer.bindVertexBuffers(VERTEX_BUFFER_BIND_ID, models.model.vertices.buffer, offsets);
		offScreenCmdBuffer.bindIndexBuffer(models.model.indices.buffer, 0, vk::IndexType::eUint32);
		offScreenCmdBuffer.drawIndexed(models.model.indexCount, 3, 0, 0, 0);

		offScreenCmdBuffer.endRenderPass();

		offScreenCmdBuffer.end();
	}

	void reBuildCommandBuffers()
	{
		if (!checkCommandBuffers())
		{
			destroyCommandBuffers();
			createCommandBuffers();
		}
		buildCommandBuffers();
		buildDeferredCommandBuffer();
	}

	void buildCommandBuffers()
	{
		vk::CommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();

		vk::ClearValue clearValues[2];
		clearValues[0].color = vk::ClearColorValue{ std::array<float, 4>{ 1.0f, 1.0f, 1.0f, 0.0f } };
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

			//std::vector<vk::DeviceSize> offsets = { 0 };
			drawCmdBuffers[i].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayouts.deferred, 0, descriptorSet, nullptr);

			if (debugDisplay)
			{
				drawCmdBuffers[i].bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.debug);
				vkCmdDraw(drawCmdBuffers[i], 3, 1, 0, 0);
				// Move viewport to display final composition in lower right corner
				viewport.x = viewport.width * 0.5f;
				viewport.y = viewport.height * 0.5f;
				viewport.width = (float)width * 0.5f;
				viewport.height = (float)height * 0.5f;
				drawCmdBuffers[i].setViewport(0, viewport);
			}

			camera.updateAspectRatio((float)viewport.width / (float)viewport.height);

			// Final composition as full screen quad
			drawCmdBuffers[i].bindPipeline(vk::PipelineBindPoint::eGraphics, useMSAA ? pipelines.deferred : pipelines.deferredNoMSAA);
			vkCmdDraw(drawCmdBuffers[i], 3, 1, 0, 0);

			drawCmdBuffers[i].endRenderPass();

			drawCmdBuffers[i].end();
		}
	}

	void loadAssets()
	{
		models.model.loadFromFile(getAssetPath() + "models/armor/armor.dae", vertexLayout, 1.0f, vulkanDevice, queue);

		vks::ModelCreateInfo modelCreateInfo;
		modelCreateInfo.scale = glm::vec3(15.0f);
		modelCreateInfo.uvscale = glm::vec2(8.0f, 8.0f);
		modelCreateInfo.center = glm::vec3(0.0f, 2.3f, 0.0f);
		models.floor.loadFromFile(getAssetPath() + "models/openbox.dae", vertexLayout, &modelCreateInfo, vulkanDevice, queue);

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
		textures.floor.colorMap.loadFromFile(getAssetPath() + "textures/stonefloor02_color" + texFormatSuffix + ".ktx", texFormat, vulkanDevice, queue);
		textures.floor.normalMap.loadFromFile(getAssetPath() + "textures/stonefloor02_normal" + texFormatSuffix + ".ktx", texFormat, vulkanDevice, queue);
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
			vks::initializers::descriptorPoolSize(vk::DescriptorType::eUniformBuffer, 8),
			vks::initializers::descriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, 9)
		};

		vk::DescriptorPoolCreateInfo descriptorPoolInfo =
			vks::initializers::descriptorPoolCreateInfo(
				static_cast<uint32_t>(poolSizes.size()),
				poolSizes.data(),
				3);

		descriptorPool = device.createDescriptorPool(descriptorPoolInfo);
	}

	void setupDescriptorSetLayout()
	{
		// Deferred shading layout
		std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings =
		{
			// Binding 0 : Vertex shader uniform buffer
			vks::initializers::descriptorSetLayoutBinding(
				vk::DescriptorType::eUniformBuffer,
				vk::ShaderStageFlagBits::eVertex,
				0),
			// Binding 1 : Position texture target / Scene colormap
			vks::initializers::descriptorSetLayoutBinding(
				vk::DescriptorType::eCombinedImageSampler,
				vk::ShaderStageFlagBits::eFragment,
				1),
			// Binding 2 : Normals texture target
			vks::initializers::descriptorSetLayoutBinding(
				vk::DescriptorType::eCombinedImageSampler,
				vk::ShaderStageFlagBits::eFragment,
				2),
			// Binding 3 : Albedo texture target
			vks::initializers::descriptorSetLayoutBinding(
				vk::DescriptorType::eCombinedImageSampler,
				vk::ShaderStageFlagBits::eFragment,
				3),
			// Binding 4 : Fragment shader uniform buffer
			vks::initializers::descriptorSetLayoutBinding(
				vk::DescriptorType::eUniformBuffer,
				vk::ShaderStageFlagBits::eFragment,
				4),
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
				colorSampler,
				offScreenFrameBuf.position.view,
				vk::ImageLayout::eShaderReadOnlyOptimal);

		vk::DescriptorImageInfo texDescriptorNormal =
			vks::initializers::descriptorImageInfo(
				colorSampler,
				offScreenFrameBuf.normal.view,
				vk::ImageLayout::eShaderReadOnlyOptimal);

		vk::DescriptorImageInfo texDescriptorAlbedo =
			vks::initializers::descriptorImageInfo(
				colorSampler,
				offScreenFrameBuf.albedo.view,
				vk::ImageLayout::eShaderReadOnlyOptimal);

		writeDescriptorSets = {
			// Binding 0 : Vertex shader uniform buffer
			vks::initializers::writeDescriptorSet(
			descriptorSet,
				vk::DescriptorType::eUniformBuffer,
				0,
				&uniformBuffers.vsFullScreen.descriptor),
			// Binding 1 : Position texture target
			vks::initializers::writeDescriptorSet(
				descriptorSet,
				vk::DescriptorType::eCombinedImageSampler,
				1,
				&texDescriptorPosition),
			// Binding 2 : Normals texture target
			vks::initializers::writeDescriptorSet(
				descriptorSet,
				vk::DescriptorType::eCombinedImageSampler,
				2,
				&texDescriptorNormal),
			// Binding 3 : Albedo texture target
			vks::initializers::writeDescriptorSet(
				descriptorSet,
				vk::DescriptorType::eCombinedImageSampler,
				3,
				&texDescriptorAlbedo),
			// Binding 4 : Fragment shader uniform buffer
			vks::initializers::writeDescriptorSet(
				descriptorSet,
				vk::DescriptorType::eUniformBuffer,
				4,
				&uniformBuffers.fsLights.descriptor),
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

		// Backbround
		descriptorSets.floor = device.allocateDescriptorSets(allocInfo)[0];
		writeDescriptorSets =
		{
			// Binding 0: Vertex shader uniform buffer
			vks::initializers::writeDescriptorSet(
				descriptorSets.floor,
				vk::DescriptorType::eUniformBuffer,
				0,
				&uniformBuffers.vsOffscreen.descriptor),
			// Binding 1: Color map
			vks::initializers::writeDescriptorSet(
				descriptorSets.floor,
				vk::DescriptorType::eCombinedImageSampler,
				1,
				&textures.floor.colorMap.descriptor),
			// Binding 2: Normal map
			vks::initializers::writeDescriptorSet(
				descriptorSets.floor,
				vk::DescriptorType::eCombinedImageSampler,
				2,
				&textures.floor.normalMap.descriptor)
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

		vk::GraphicsPipelineCreateInfo pipelineCreateInfo =
			vks::initializers::pipelineCreateInfo(
				pipelineLayouts.deferred,
				renderPass);

		pipelineCreateInfo.pInputAssemblyState = &inputAssemblyState;
		pipelineCreateInfo.pRasterizationState = &rasterizationState;
		pipelineCreateInfo.pColorBlendState = &colorBlendState;
		pipelineCreateInfo.pMultisampleState = &multisampleState;
		pipelineCreateInfo.pViewportState = &viewportState;
		pipelineCreateInfo.pDepthStencilState = &depthStencilState;
		pipelineCreateInfo.pDynamicState = &dynamicState;
		pipelineCreateInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
		pipelineCreateInfo.pStages = shaderStages.data();

		// Deferred

		// Empty vertex input state, quads are generated by the vertex shader
		vk::PipelineVertexInputStateCreateInfo emptyInputState = vks::initializers::pipelineVertexInputStateCreateInfo();
		pipelineCreateInfo.pVertexInputState = &emptyInputState;
		pipelineCreateInfo.layout = pipelineLayouts.deferred;

		// Use specialization constants to pass number of samples to the shader (used for MSAA resolve)
		vk::SpecializationMapEntry specializationEntry{};
		specializationEntry.constantID = 0;
		specializationEntry.offset = 0;
		specializationEntry.size = sizeof(uint32_t);
		
		uint32_t specializationData = uint32_t(SAMPLE_COUNT);

		vk::SpecializationInfo specializationInfo;
		specializationInfo.mapEntryCount = 1;
		specializationInfo.pMapEntries = &specializationEntry;
		specializationInfo.dataSize = sizeof(specializationData);
		specializationInfo.pData = &specializationData;

		// With MSAA
		shaderStages[0] = loadShader(getAssetPath() + "shaders/deferredmultisampling/deferred.vert.spv", vk::ShaderStageFlagBits::eVertex);
		shaderStages[1] = loadShader(getAssetPath() + "shaders/deferredmultisampling/deferred.frag.spv", vk::ShaderStageFlagBits::eFragment);
		shaderStages[1].pSpecializationInfo = &specializationInfo;
		pipelines.deferred = device.createGraphicsPipelines(pipelineCache, pipelineCreateInfo)[0];

		// No MSAA (1 sample)
		specializationData = 1;
		pipelines.deferredNoMSAA = device.createGraphicsPipelines(pipelineCache, pipelineCreateInfo)[0];

		// Debug display pipeline
		shaderStages[0] = loadShader(getAssetPath() + "shaders/deferredmultisampling/debug.vert.spv", vk::ShaderStageFlagBits::eVertex);
		shaderStages[1] = loadShader(getAssetPath() + "shaders/deferredmultisampling/debug.frag.spv", vk::ShaderStageFlagBits::eFragment);
		pipelines.debug = device.createGraphicsPipelines(pipelineCache, pipelineCreateInfo)[0];
		
		// Offscreen scene rendering pipeline
		pipelineCreateInfo.pVertexInputState = &vertices.inputState;

		shaderStages[0] = loadShader(getAssetPath() + "shaders/deferredmultisampling/mrt.vert.spv", vk::ShaderStageFlagBits::eVertex);
		shaderStages[1] = loadShader(getAssetPath() + "shaders/deferredmultisampling/mrt.frag.spv", vk::ShaderStageFlagBits::eFragment);

		//rasterizationState.polygonMode = vk::PolygonMode::eLine;
		//rasterizationState.lineWidth = 2.0f;
		multisampleState.rasterizationSamples = SAMPLE_COUNT;
		multisampleState.alphaToCoverageEnable = VK_TRUE;

		// Separate render pass
		pipelineCreateInfo.renderPass = offScreenFrameBuf.renderPass;

		// Separate layout
		pipelineCreateInfo.layout = pipelineLayouts.offscreen;

		// Blend attachment states required for all color attachments
		// This is important, as color write mask will otherwise be 0x0 and you
		// won't see anything rendered to the attachment
		vk::PipelineColorBlendAttachmentState tmp = vks::initializers::pipelineColorBlendAttachmentState(
			vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA, 
			VK_FALSE);
		std::array<vk::PipelineColorBlendAttachmentState, 3> blendAttachmentStates = {
			tmp, tmp, tmp
		};

		colorBlendState.attachmentCount = static_cast<uint32_t>(blendAttachmentStates.size());
		colorBlendState.pAttachments = blendAttachmentStates.data();

		pipelines.offscreen = device.createGraphicsPipelines(pipelineCache, pipelineCreateInfo)[0];

		multisampleState.sampleShadingEnable = VK_TRUE;
		multisampleState.minSampleShading = 0.25f;
		pipelines.offscreenSampleShading = device.createGraphicsPipelines(pipelineCache, pipelineCreateInfo)[0];
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

		// Map persistent
		uniformBuffers.vsFullScreen.map();
		uniformBuffers.vsOffscreen.map();
		uniformBuffers.fsLights.map();

		// Init some values
		uboOffscreenVS.instancePos[0] = glm::vec4(0.0f);
		uboOffscreenVS.instancePos[1] = glm::vec4(-4.0f, 0.0, -4.0f, 0.0f);
		uboOffscreenVS.instancePos[2] = glm::vec4(4.0f, 0.0, -4.0f, 0.0f);

		// Update
		updateUniformBuffersScreen();
		updateUniformBufferDeferredMatrices();
		updateUniformBufferDeferredLights();
	}

	void updateUniformBuffersScreen()
	{
		if (debugDisplay)
		{
			uboVS.projection = glm::ortho(0.0f, 2.0f, 0.0f, 2.0f, -1.0f, 1.0f);
		} 
		else
		{
			uboVS.projection = glm::ortho(0.0f, 1.0f, 0.0f, 1.0f, -1.0f, 1.0f);
		}
		uboVS.model = glm::mat4();

		memcpy(uniformBuffers.vsFullScreen.mapped, &uboVS, sizeof(uboVS));
	}

	void updateUniformBufferDeferredMatrices()
	{
		uboOffscreenVS.projection = glm::perspective(glm::radians(45.0f), (float)width / (float)height, 0.1f, 256.0f);
		uboOffscreenVS.view = glm::translate(glm::mat4(), glm::vec3(0.0f, 0.0f, zoom));

		uboOffscreenVS.model = glm::mat4();
		uboOffscreenVS.model = glm::translate(glm::mat4(), glm::vec3(0.0f, 0.25f, 0.0f) + cameraPos);
		uboOffscreenVS.model = glm::rotate(uboOffscreenVS.model, glm::radians(rotation.x), glm::vec3(1.0f, 0.0f, 0.0f));
		uboOffscreenVS.model = glm::rotate(uboOffscreenVS.model, glm::radians(rotation.y), glm::vec3(0.0f, 1.0f, 0.0f));
		uboOffscreenVS.model = glm::rotate(uboOffscreenVS.model, glm::radians(rotation.z), glm::vec3(0.0f, 0.0f, 1.0f));

		uboOffscreenVS.projection = camera.matrices.perspective;
		uboOffscreenVS.view = camera.matrices.view;
		uboOffscreenVS.model = glm::mat4();

		memcpy(uniformBuffers.vsOffscreen.mapped, &uboOffscreenVS, sizeof(uboOffscreenVS));
	}

	// Update fragment shader light position uniform block
	void updateUniformBufferDeferredLights()
	{
		// White
		uboFragmentLights.lights[0].position = glm::vec4(0.0f, 0.0f, 1.0f, 0.0f);
		uboFragmentLights.lights[0].color = glm::vec3(1.5f);
		uboFragmentLights.lights[0].radius = 15.0f * 0.25f;
		// Red
		uboFragmentLights.lights[1].position = glm::vec4(-2.0f, 0.0f, 0.0f, 0.0f);
		uboFragmentLights.lights[1].color = glm::vec3(1.0f, 0.0f, 0.0f);
		uboFragmentLights.lights[1].radius = 15.0f;
		// Blue
		uboFragmentLights.lights[2].position = glm::vec4(2.0f, 1.0f, 0.0f, 0.0f);
		uboFragmentLights.lights[2].color = glm::vec3(0.0f, 0.0f, 2.5f);
		uboFragmentLights.lights[2].radius = 5.0f;
		// Yellow
		uboFragmentLights.lights[3].position = glm::vec4(0.0f, 0.9f, 0.5f, 0.0f);
		uboFragmentLights.lights[3].color = glm::vec3(1.0f, 1.0f, 0.0f);
		uboFragmentLights.lights[3].radius = 2.0f;
		// Green
		uboFragmentLights.lights[4].position = glm::vec4(0.0f, 0.5f, 0.0f, 0.0f);
		uboFragmentLights.lights[4].color = glm::vec3(0.0f, 1.0f, 0.2f);
		uboFragmentLights.lights[4].radius = 5.0f;
		// Yellow
		uboFragmentLights.lights[5].position = glm::vec4(0.0f, 1.0f, 0.0f, 0.0f);
		uboFragmentLights.lights[5].color = glm::vec3(1.0f, 0.7f, 0.3f);
		uboFragmentLights.lights[5].radius = 25.0f;

		uboFragmentLights.lights[0].position.x = sin(glm::radians(360.0f * timer)) * 5.0f;
		uboFragmentLights.lights[0].position.z = cos(glm::radians(360.0f * timer)) * 5.0f;

		uboFragmentLights.lights[1].position.x = -4.0f + sin(glm::radians(360.0f * timer) + 45.0f) * 2.0f;
		uboFragmentLights.lights[1].position.z =  0.0f + cos(glm::radians(360.0f * timer) + 45.0f) * 2.0f;

		uboFragmentLights.lights[2].position.x = 4.0f + sin(glm::radians(360.0f * timer)) * 2.0f;
		uboFragmentLights.lights[2].position.z = 0.0f + cos(glm::radians(360.0f * timer)) * 2.0f;

		uboFragmentLights.lights[4].position.x = 0.0f + sin(glm::radians(360.0f * timer + 90.0f)) * 5.0f;
		uboFragmentLights.lights[4].position.z = 0.0f - cos(glm::radians(360.0f * timer + 45.0f)) * 5.0f;

		uboFragmentLights.lights[5].position.x = 0.0f + sin(glm::radians(-360.0f * timer + 135.0f)) * 10.0f;
		uboFragmentLights.lights[5].position.z = 0.0f - cos(glm::radians(-360.0f * timer - 45.0f)) * 10.0f;

		// Current view position
		uboFragmentLights.viewPos = glm::vec4(camera.position, 0.0f) * glm::vec4(-1.0f, 1.0f, -1.0f, 1.0f);

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
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &offScreenCmdBuffer;
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
		setupVertexDescriptions();
		prepareOffscreenFramebuffer();
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
		uboFragmentLights.windowSize = glm::ivec2(width, height);
	}

	void toggleDebugDisplay()
	{
		debugDisplay = !debugDisplay;
		reBuildCommandBuffers();
		updateUniformBuffersScreen();
	}

	virtual void keyPressed(uint32_t keyCode)
	{
		switch (keyCode)
		{
		case KEY_F2:
			useMSAA = !useMSAA;
			reBuildCommandBuffers();
			break;
		case KEY_F3:
			useSampleShading = !useSampleShading;
			reBuildCommandBuffers();
			break;
		case KEY_F4:
		case GAMEPAD_BUTTON_A:
			toggleDebugDisplay();
			updateTextOverlay();
			break;
		}
	}

	virtual void getOverlayText(VulkanTextOverlay *textOverlay)
	{
#if defined(__ANDROID__)
		textOverlay->addText("Press \"Button A\" to toggle debug display", 5.0f, 85.0f, VulkanTextOverlay::alignLeft);
#else
		textOverlay->addText("MSAA (\"F2\"): " + std::to_string(useMSAA), 5.0f, 85.0f, VulkanTextOverlay::alignLeft);
		textOverlay->addText("Sample Shading (\"F3\"): " + std::to_string(useSampleShading), 5.0f, 105.0f, VulkanTextOverlay::alignLeft);
		textOverlay->addText("G-Buffers (\"F4\")", 5.0f, 125.0f, VulkanTextOverlay::alignLeft);
#endif
		// Render targets
		if (debugDisplay)
		{
			textOverlay->addText("World space position", (float)width * 0.25f, (float)height * 0.5f - 25.0f, VulkanTextOverlay::alignCenter);
			textOverlay->addText("World space normals", (float)width * 0.75f, (float)height * 0.5f - 25.0f, VulkanTextOverlay::alignCenter);
			textOverlay->addText("Albedo", (float)width * 0.25f, (float)height - 25.0f, VulkanTextOverlay::alignCenter);
			textOverlay->addText("Final image", (float)width * 0.75f, (float)height - 25.0f, VulkanTextOverlay::alignCenter);
		}
	}
};

VULKAN_EXAMPLE_MAIN()
