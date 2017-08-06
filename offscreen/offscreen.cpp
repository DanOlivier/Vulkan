/*
* Vulkan Example - Offscreen rendering using a separate framebuffer
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
#include "VulkanTexture.hpp"
#include "VulkanModel.hpp"
#include "VulkanBuffer.hpp"

#define VERTEX_BUFFER_BIND_ID 0
#define ENABLE_VALIDATION false

// Offscreen frame buffer properties
#define FB_DIM 512
#define FB_COLOR_FORMAT vk::Format::eR8G8B8A8Unorm

class VulkanExample : public VulkanExampleBase
{
public:
	bool debugDisplay = false;

	struct {
		vks::Texture2D colorMap;
	} textures;
	
	// Vertex layout for the models
	vks::VertexLayout vertexLayout = vks::VertexLayout({
		vks::VERTEX_COMPONENT_POSITION,
		vks::VERTEX_COMPONENT_UV,
		vks::VERTEX_COMPONENT_COLOR,
		vks::VERTEX_COMPONENT_NORMAL,
	});

	struct {
		vks::Model example;
		vks::Model quad;
		vks::Model plane;
	} models;

	struct {
		vk::PipelineVertexInputStateCreateInfo inputState;
		std::vector<vk::VertexInputBindingDescription> bindingDescriptions;
		std::vector<vk::VertexInputAttributeDescription> attributeDescriptions;
	} vertices;

	struct {
		vks::Buffer vsShared;
		vks::Buffer vsMirror;
		vks::Buffer vsOffScreen;
		vks::Buffer vsDebugQuad;
	} uniformBuffers;

	struct UBO {
		glm::mat4 projection;
		glm::mat4 model;
		glm::vec4 lightPos = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
	} uboShared;

	struct {
		vk::Pipeline debug;
		vk::Pipeline shaded;
		vk::Pipeline shadedOffscreen;
		vk::Pipeline mirror;
	} pipelines;

	struct {
		vk::PipelineLayout textured;
		vk::PipelineLayout shaded;
	} pipelineLayouts;

	struct {
		vk::DescriptorSet offscreen;
		vk::DescriptorSet mirror;
		vk::DescriptorSet model;
		vk::DescriptorSet debugQuad;
	} descriptorSets;

	struct {
		vk::DescriptorSetLayout textured;
		vk::DescriptorSetLayout shaded;
	} descriptorSetLayouts;

	// Framebuffer for offscreen rendering
	struct FrameBufferAttachment {
		vk::Image image;
		vk::DeviceMemory mem;
		vk::ImageView view;
	};
	struct OffscreenPass {
		int32_t width, height;
		vk::Framebuffer frameBuffer;		
		FrameBufferAttachment color, depth;
		vk::RenderPass renderPass;
		vk::Sampler sampler;
		vk::DescriptorImageInfo descriptor;
		vk::CommandBuffer commandBuffer;
		// Semaphore used to synchronize between offscreen and final scene render pass
		vk::Semaphore semaphore;
	} offscreenPass;

	glm::vec3 meshPos = glm::vec3(0.0f, -1.5f, 0.0f);
	glm::vec3 meshRot = glm::vec3(0.0f);

	VulkanExample() : VulkanExampleBase(ENABLE_VALIDATION)
	{
		zoom = -6.0f;
		rotation = { -2.5f, 0.0f, 0.0f };
		cameraPos = { 0.0f, 1.0f, 0.0f };
		timerSpeed *= 0.25f;
		enableTextOverlay = true;
		title = "Vulkan Example - Offscreen rendering";
		enabledFeatures.shaderClipDistance = VK_TRUE;
	}

	~VulkanExample()
	{
		// Clean up used Vulkan resources 
		// Note : Inherited destructor cleans up resources stored in base class

		// Textures
		textures.colorMap.destroy();

		// Frame buffer

		// Color attachment
		device.destroyImageView(offscreenPass.color.view);
		device.destroyImage(offscreenPass.color.image);
		device.freeMemory(offscreenPass.color.mem);

		// Depth attachment
		device.destroyImageView(offscreenPass.depth.view);
		device.destroyImage(offscreenPass.depth.image);
		device.freeMemory(offscreenPass.depth.mem);

		device.destroyRenderPass(offscreenPass.renderPass);
		device.destroySampler(offscreenPass.sampler);
		device.destroyFramebuffer(offscreenPass.frameBuffer);

		device.destroyPipeline(pipelines.debug);
		device.destroyPipeline(pipelines.shaded);
		device.destroyPipeline(pipelines.shadedOffscreen);
		device.destroyPipeline(pipelines.mirror);

		device.destroyPipelineLayout(pipelineLayouts.textured);
		device.destroyPipelineLayout(pipelineLayouts.shaded);

		device.destroyDescriptorSetLayout(descriptorSetLayouts.shaded);
		device.destroyDescriptorSetLayout(descriptorSetLayouts.textured);

		// Models
		models.example.destroy();
		models.quad.destroy();
		models.plane.destroy();

		// Uniform buffers
		uniformBuffers.vsShared.destroy();
		uniformBuffers.vsMirror.destroy();
		uniformBuffers.vsOffScreen.destroy();
		uniformBuffers.vsDebugQuad.destroy();

		device.freeCommandBuffers(cmdPool, offscreenPass.commandBuffer);
		device.destroySemaphore(offscreenPass.semaphore);
	}

	// Setup the offscreen framebuffer for rendering the mirrored scene
	// The color attachment of this framebuffer will then be used to sample from in the fragment shader of the final pass
	void prepareOffscreen()
	{
		offscreenPass.width = FB_DIM;
		offscreenPass.height = FB_DIM;

		// Find a suitable depth format
		vk::Format fbDepthFormat;
		vk::Bool32 validDepthFormat = vks::tools::getSupportedDepthFormat(physicalDevice, &fbDepthFormat);
		assert(validDepthFormat);

		// Color attachment
		vk::ImageCreateInfo image = vks::initializers::imageCreateInfo();
		image.imageType = vk::ImageType::e2D;
		image.format = FB_COLOR_FORMAT;
		image.extent.width = offscreenPass.width;
		image.extent.height = offscreenPass.height;
		image.extent.depth = 1;
		image.mipLevels = 1;
		image.arrayLayers = 1;
		image.samples = vk::SampleCountFlagBits::e1;
		image.tiling = vk::ImageTiling::eOptimal;
		// We will sample directly from the color attachment
		image.usage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled;

		vk::MemoryAllocateInfo memAlloc = vks::initializers::memoryAllocateInfo();
		vk::MemoryRequirements memReqs;

		offscreenPass.color.image = device.createImage(image);
		memReqs = device.getImageMemoryRequirements(offscreenPass.color.image);
		memAlloc.allocationSize = memReqs.size;
		memAlloc.memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);
		offscreenPass.color.mem = device.allocateMemory(memAlloc);
		device.bindImageMemory(offscreenPass.color.image, offscreenPass.color.mem, 0);

		vk::ImageViewCreateInfo colorImageView = vks::initializers::imageViewCreateInfo();
		colorImageView.viewType = vk::ImageViewType::e2D;
		colorImageView.format = FB_COLOR_FORMAT;
		//colorImageView.subresourceRange = {};
		colorImageView.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
		colorImageView.subresourceRange.baseMipLevel = 0;
		colorImageView.subresourceRange.levelCount = 1;
		colorImageView.subresourceRange.baseArrayLayer = 0;
		colorImageView.subresourceRange.layerCount = 1;
		colorImageView.image = offscreenPass.color.image;
		offscreenPass.color.view = device.createImageView(colorImageView);

		// Create sampler to sample from the attachment in the fragment shader
		vk::SamplerCreateInfo samplerInfo = vks::initializers::samplerCreateInfo();
		samplerInfo.magFilter = vk::Filter::eLinear;
		samplerInfo.minFilter = vk::Filter::eLinear;
		samplerInfo.mipmapMode = vk::SamplerMipmapMode::eLinear;
		samplerInfo.addressModeU = vk::SamplerAddressMode::eClampToEdge;
		samplerInfo.addressModeV = samplerInfo.addressModeU;
		samplerInfo.addressModeW = samplerInfo.addressModeU;
		samplerInfo.mipLodBias = 0.0f;
		samplerInfo.maxAnisotropy = 1.0f;
		samplerInfo.minLod = 0.0f;
		samplerInfo.maxLod = 1.0f;
		samplerInfo.borderColor = vk::BorderColor::eFloatOpaqueWhite;
		offscreenPass.sampler = device.createSampler(samplerInfo);

		// Depth stencil attachment
		image.format = fbDepthFormat;
		image.usage = vk::ImageUsageFlagBits::eDepthStencilAttachment;

		offscreenPass.depth.image = device.createImage(image);
		memReqs = device.getImageMemoryRequirements(offscreenPass.depth.image);
		memAlloc.allocationSize = memReqs.size;
		memAlloc.memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);
		offscreenPass.depth.mem = device.allocateMemory(memAlloc);
		device.bindImageMemory(offscreenPass.depth.image, offscreenPass.depth.mem, 0);

		vk::ImageViewCreateInfo depthStencilView = vks::initializers::imageViewCreateInfo();
		depthStencilView.viewType = vk::ImageViewType::e2D;
		depthStencilView.format = fbDepthFormat;
		//depthStencilView.flags = 0;
		//depthStencilView.subresourceRange = {};
		depthStencilView.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil;
		depthStencilView.subresourceRange.baseMipLevel = 0;
		depthStencilView.subresourceRange.levelCount = 1;
		depthStencilView.subresourceRange.baseArrayLayer = 0;
		depthStencilView.subresourceRange.layerCount = 1;
		depthStencilView.image = offscreenPass.depth.image;
		offscreenPass.depth.view = device.createImageView(depthStencilView);

		// Create a separate render pass for the offscreen rendering as it may differ from the one used for scene rendering

		std::array<vk::AttachmentDescription, 2> attchmentDescriptions = {};
		// Color attachment
		attchmentDescriptions[0].format = FB_COLOR_FORMAT;
		attchmentDescriptions[0].samples = vk::SampleCountFlagBits::e1;
		attchmentDescriptions[0].loadOp = vk::AttachmentLoadOp::eClear;
		attchmentDescriptions[0].storeOp = vk::AttachmentStoreOp::eStore;
		attchmentDescriptions[0].stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
		attchmentDescriptions[0].stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
		attchmentDescriptions[0].initialLayout = vk::ImageLayout::eUndefined;
		attchmentDescriptions[0].finalLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
		// Depth attachment
		attchmentDescriptions[1].format = fbDepthFormat;
		attchmentDescriptions[1].samples = vk::SampleCountFlagBits::e1;
		attchmentDescriptions[1].loadOp = vk::AttachmentLoadOp::eClear;
		attchmentDescriptions[1].storeOp = vk::AttachmentStoreOp::eDontCare;
		attchmentDescriptions[1].stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
		attchmentDescriptions[1].stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
		attchmentDescriptions[1].initialLayout = vk::ImageLayout::eUndefined;
		attchmentDescriptions[1].finalLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal;

		vk::AttachmentReference colorReference = { 0, vk::ImageLayout::eColorAttachmentOptimal };
		vk::AttachmentReference depthReference = { 1, vk::ImageLayout::eDepthStencilAttachmentOptimal };

		vk::SubpassDescription subpassDescription = {};
		subpassDescription.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
		subpassDescription.colorAttachmentCount = 1;
		subpassDescription.pColorAttachments = &colorReference;
		subpassDescription.pDepthStencilAttachment = &depthReference;

		// Use subpass dependencies for layout transitions
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

		// Create the actual renderpass
		vk::RenderPassCreateInfo renderPassInfo = {};

		renderPassInfo.attachmentCount = static_cast<uint32_t>(attchmentDescriptions.size());
		renderPassInfo.pAttachments = attchmentDescriptions.data();
		renderPassInfo.subpassCount = 1;
		renderPassInfo.pSubpasses = &subpassDescription;
		renderPassInfo.dependencyCount = static_cast<uint32_t>(dependencies.size());
		renderPassInfo.pDependencies = dependencies.data();

		offscreenPass.renderPass = device.createRenderPass(renderPassInfo);

		vk::ImageView attachments[2];
		attachments[0] = offscreenPass.color.view;
		attachments[1] = offscreenPass.depth.view;

		vk::FramebufferCreateInfo fbufCreateInfo = vks::initializers::framebufferCreateInfo();
		fbufCreateInfo.renderPass = offscreenPass.renderPass;
		fbufCreateInfo.attachmentCount = 2;
		fbufCreateInfo.pAttachments = attachments;
		fbufCreateInfo.width = offscreenPass.width;
		fbufCreateInfo.height = offscreenPass.height;
		fbufCreateInfo.layers = 1;

		offscreenPass.frameBuffer = device.createFramebuffer(fbufCreateInfo);

		// Fill a descriptor for later use in a descriptor set 
		offscreenPass.descriptor.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
		offscreenPass.descriptor.imageView = offscreenPass.color.view;
		offscreenPass.descriptor.sampler = offscreenPass.sampler;
	}

	// Sets up the command buffer that renders the scene to the offscreen frame buffer
	void buildOffscreenCommandBuffer()
	{
		if (!offscreenPass.commandBuffer)
		{
			offscreenPass.commandBuffer = VulkanExampleBase::createCommandBuffer(vk::CommandBufferLevel::ePrimary, false);
		}
		if (!offscreenPass.semaphore)
		{
			// Create a semaphore used to synchronize offscreen rendering and usage
			vk::SemaphoreCreateInfo semaphoreCreateInfo = vks::initializers::semaphoreCreateInfo();
			offscreenPass.semaphore = device.createSemaphore(semaphoreCreateInfo);
		}

		vk::CommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();

		vk::ClearValue clearValues[2];
		clearValues[0].color = vk::ClearColorValue{ std::array<float, 4>{ 0.0f, 0.0f, 0.0f, 0.0f } };
		clearValues[1].depthStencil = vk::ClearDepthStencilValue{ 1.0f, 0 };

		vk::RenderPassBeginInfo renderPassBeginInfo = vks::initializers::renderPassBeginInfo();
		renderPassBeginInfo.renderPass = offscreenPass.renderPass;
		renderPassBeginInfo.framebuffer = offscreenPass.frameBuffer;
		renderPassBeginInfo.renderArea.extent.width = offscreenPass.width;
		renderPassBeginInfo.renderArea.extent.height = offscreenPass.height;
		renderPassBeginInfo.clearValueCount = 2;
		renderPassBeginInfo.pClearValues = clearValues;

		offscreenPass.commandBuffer.begin(cmdBufInfo);

		offscreenPass.commandBuffer.beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);

		vk::Viewport viewport = vks::initializers::viewport((float)offscreenPass.width, (float)offscreenPass.height, 0.0f, 1.0f);
		offscreenPass.commandBuffer.setViewport(0, viewport);

		vk::Rect2D scissor = vks::initializers::rect2D(offscreenPass.width, offscreenPass.height, 0, 0);
		offscreenPass.commandBuffer.setScissor(0, scissor);

		std::vector<vk::DeviceSize> offsets = { 0 };

		// Mirrored scene
		offscreenPass.commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayouts.shaded, 0, descriptorSets.offscreen, nullptr);
		offscreenPass.commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.shadedOffscreen);
		offscreenPass.commandBuffer.bindVertexBuffers(VERTEX_BUFFER_BIND_ID, models.example.vertices.buffer, offsets);
		offscreenPass.commandBuffer.bindIndexBuffer(models.example.indices.buffer, 0, vk::IndexType::eUint32);
		offscreenPass.commandBuffer.drawIndexed(models.example.indexCount, 1, 0, 0, 0);

		offscreenPass.commandBuffer.endRenderPass();

		offscreenPass.commandBuffer.end();
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

			vk::Rect2D scissor = vks::initializers::rect2D(width, height,	0, 0);
			drawCmdBuffers[i].setScissor(0, scissor);

			std::vector<vk::DeviceSize> offsets = { 0 };

			if (debugDisplay)
			{
				drawCmdBuffers[i].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayouts.textured, 0, descriptorSets.debugQuad, nullptr);
				drawCmdBuffers[i].bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.debug);
				drawCmdBuffers[i].bindVertexBuffers(VERTEX_BUFFER_BIND_ID, models.quad.vertices.buffer, offsets);
				drawCmdBuffers[i].bindIndexBuffer(models.quad.indices.buffer, 0, vk::IndexType::eUint32);
				drawCmdBuffers[i].drawIndexed(models.quad.indexCount, 1, 0, 0, 0);
			}

			// Scene
			drawCmdBuffers[i].bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.debug);

			// Reflection plane
			drawCmdBuffers[i].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayouts.textured, 0, descriptorSets.mirror, nullptr);
			drawCmdBuffers[i].bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.mirror);

			drawCmdBuffers[i].bindVertexBuffers(VERTEX_BUFFER_BIND_ID, models.plane.vertices.buffer, offsets);
			drawCmdBuffers[i].bindIndexBuffer(models.plane.indices.buffer, 0, vk::IndexType::eUint32);
			drawCmdBuffers[i].drawIndexed(models.plane.indexCount, 1, 0, 0, 0);

			// Model
			drawCmdBuffers[i].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayouts.shaded, 0, descriptorSets.model, nullptr);
			drawCmdBuffers[i].bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.shaded);

			drawCmdBuffers[i].bindVertexBuffers(VERTEX_BUFFER_BIND_ID, models.example.vertices.buffer, offsets);
			drawCmdBuffers[i].bindIndexBuffer(models.example.indices.buffer, 0, vk::IndexType::eUint32);
			drawCmdBuffers[i].drawIndexed(models.example.indexCount, 1, 0, 0, 0);

			drawCmdBuffers[i].endRenderPass();

			drawCmdBuffers[i].end();
		}
	}

	void loadAssets()
	{
		models.plane.loadFromFile(getAssetPath() + "models/plane.obj", vertexLayout, 0.5f, vulkanDevice, queue);
		models.example.loadFromFile(getAssetPath() + "models/chinesedragon.dae", vertexLayout, 0.3f, vulkanDevice, queue);

		// Textures
		if (vulkanDevice->features.textureCompressionBC) {
			textures.colorMap.loadFromFile(getAssetPath() + "textures/darkmetal_bc3_unorm.ktx", vk::Format::eBc3UnormBlock, vulkanDevice, queue);
		}
		else if (vulkanDevice->features.textureCompressionASTC_LDR) {
			textures.colorMap.loadFromFile(getAssetPath() + "textures/darkmetal_astc_8x8_unorm.ktx", vk::Format::eAstc8x8UnormBlock, vulkanDevice, queue);
		}
		else if (vulkanDevice->features.textureCompressionETC2) {
			textures.colorMap.loadFromFile(getAssetPath() + "textures/darkmetal_etc2_unorm.ktx", vk::Format::eEtc2R8G8B8UnormBlock, vulkanDevice, queue);
		}
		else {
			vks::tools::exitFatal("Device does not support any compressed texture format!", "Error");
		}
	}

	void generateQuad()
	{
		// Setup vertices for a single uv-mapped quad
		struct Vertex {
			float pos[3];
			float uv[2];
			float col[3];
			float normal[3];
		};

#define QUAD_COLOR_NORMAL { 1.0f, 1.0f, 1.0f }, { 0.0f, 0.0f, 1.0f }
		std::vector<Vertex> vertexBuffer =
		{
			{ { 1.0f, 1.0f, 0.0f },{ 1.0f, 1.0f }, QUAD_COLOR_NORMAL },
			{ { 0.0f, 1.0f, 0.0f },{ 0.0f, 1.0f }, QUAD_COLOR_NORMAL },
			{ { 0.0f, 0.0f, 0.0f },{ 0.0f, 0.0f }, QUAD_COLOR_NORMAL },
			{ { 1.0f, 0.0f, 0.0f },{ 1.0f, 0.0f }, QUAD_COLOR_NORMAL }
		};
#undef QUAD_COLOR_NORMAL

		vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eVertexBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			vertexBuffer.size() * sizeof(Vertex),
			&models.quad.vertices.buffer,
			&models.quad.vertices.memory,
			vertexBuffer.data());

		// Setup indices
		std::vector<uint32_t> indexBuffer = { 0,1,2, 2,3,0 };
		models.quad.indexCount = indexBuffer.size();

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
		vertices.attributeDescriptions.resize(4);
		// Location 0 : Position
		vertices.attributeDescriptions[0] =
			vks::initializers::vertexInputAttributeDescription(
				VERTEX_BUFFER_BIND_ID,
				0,
				vk::Format::eR32G32B32Sfloat,
				0);
		// Location 1 : Texture coordinates
		vertices.attributeDescriptions[1] =
			vks::initializers::vertexInputAttributeDescription(
				VERTEX_BUFFER_BIND_ID,
				1,
				vk::Format::eR32G32Sfloat,
				sizeof(float) * 3);
		// Location 2 : Color
		vertices.attributeDescriptions[2] =
			vks::initializers::vertexInputAttributeDescription(
				VERTEX_BUFFER_BIND_ID,
				2,
				vk::Format::eR32G32B32Sfloat,
				sizeof(float) * 5);
		// Location 3 : Normal
		vertices.attributeDescriptions[3] =
			vks::initializers::vertexInputAttributeDescription(
				VERTEX_BUFFER_BIND_ID,
				3,
				vk::Format::eR32G32B32Sfloat,
				sizeof(float) * 8);

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
			vks::initializers::descriptorPoolSize(vk::DescriptorType::eUniformBuffer, 6),
			vks::initializers::descriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, 8)
		};

		vk::DescriptorPoolCreateInfo descriptorPoolInfo =
			vks::initializers::descriptorPoolCreateInfo(
				poolSizes.size(),
				poolSizes.data(),
				5);

		descriptorPool = device.createDescriptorPool(descriptorPoolInfo);
	}

	void setupDescriptorSetLayout()
	{
		std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings;
		vk::DescriptorSetLayoutCreateInfo descriptorLayoutInfo;
		vk::PipelineLayoutCreateInfo pipelineLayoutInfo;

		// Binding 0 : Vertex shader uniform buffer
		setLayoutBindings.push_back(vks::initializers::descriptorSetLayoutBinding(
			vk::DescriptorType::eUniformBuffer,
			vk::ShaderStageFlagBits::eVertex,
			0));
		// Binding 1 : Fragment shader image sampler
		setLayoutBindings.push_back(vks::initializers::descriptorSetLayoutBinding(
			vk::DescriptorType::eCombinedImageSampler,
			vk::ShaderStageFlagBits::eFragment,
			1));
		// Binding 2 : Fragment shader image sampler
		setLayoutBindings.push_back(vks::initializers::descriptorSetLayoutBinding(
			vk::DescriptorType::eCombinedImageSampler,
			vk::ShaderStageFlagBits::eFragment,
			2));

		// Shaded layouts (only use first layout binding)
		descriptorLayoutInfo = vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings.data(), 1);
		descriptorSetLayouts.shaded = device.createDescriptorSetLayout(descriptorLayoutInfo);

		pipelineLayoutInfo = vks::initializers::pipelineLayoutCreateInfo(&descriptorSetLayouts.shaded, 1);
		pipelineLayouts.shaded = device.createPipelineLayout(pipelineLayoutInfo);

		// Textured layouts (use all layout bindings)
		descriptorLayoutInfo = vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings.data(), static_cast<uint32_t>(setLayoutBindings.size()));
		descriptorSetLayouts.textured = device.createDescriptorSetLayout(descriptorLayoutInfo);

		pipelineLayoutInfo = vks::initializers::pipelineLayoutCreateInfo(&descriptorSetLayouts.textured, 1);
		pipelineLayouts.textured = device.createPipelineLayout(pipelineLayoutInfo);
	}

	void setupDescriptorSet()
	{
		// Mirror plane descriptor set
		vk::DescriptorSetAllocateInfo allocInfo =
			vks::initializers::descriptorSetAllocateInfo(
				descriptorPool,
				&descriptorSetLayouts.textured,
				1);

		descriptorSets.mirror = device.allocateDescriptorSets(allocInfo)[0];

		std::vector<vk::WriteDescriptorSet> writeDescriptorSets =
		{
			// Binding 0 : Vertex shader uniform buffer
			vks::initializers::writeDescriptorSet(
				descriptorSets.mirror,
				vk::DescriptorType::eUniformBuffer,
				0,
				&uniformBuffers.vsMirror.descriptor),
			// Binding 1 : Fragment shader texture sampler
			vks::initializers::writeDescriptorSet(
				descriptorSets.mirror,
				vk::DescriptorType::eCombinedImageSampler,
				1,
				&offscreenPass.descriptor),
			// Binding 2 : Fragment shader texture sampler
			vks::initializers::writeDescriptorSet(
				descriptorSets.mirror,
				vk::DescriptorType::eCombinedImageSampler,
				2,
				&textures.colorMap.descriptor)
		};

		device.updateDescriptorSets(writeDescriptorSets, nullptr);

		// Debug quad
		descriptorSets.debugQuad = device.allocateDescriptorSets(allocInfo)[0];

		std::vector<vk::WriteDescriptorSet> debugQuadWriteDescriptorSets =
		{
			// Binding 0 : Vertex shader uniform buffer
			vks::initializers::writeDescriptorSet(
				descriptorSets.debugQuad,
				vk::DescriptorType::eUniformBuffer,
				0,
				&uniformBuffers.vsDebugQuad.descriptor),
			// Binding 1 : Fragment shader texture sampler
			vks::initializers::writeDescriptorSet(
				descriptorSets.debugQuad,
				vk::DescriptorType::eCombinedImageSampler,
				1,
				&offscreenPass.descriptor)
		};
		device.updateDescriptorSets(debugQuadWriteDescriptorSets, nullptr);

		// Shaded descriptor sets
		allocInfo.pSetLayouts = &descriptorSetLayouts.shaded;

		// Model
		// No texture
		descriptorSets.model = device.allocateDescriptorSets(allocInfo)[0];

		std::vector<vk::WriteDescriptorSet> modelWriteDescriptorSets =
		{
			// Binding 0 : Vertex shader uniform buffer
			vks::initializers::writeDescriptorSet(
				descriptorSets.model,
				vk::DescriptorType::eUniformBuffer,
				0,
				&uniformBuffers.vsShared.descriptor)
		};
		device.updateDescriptorSets(modelWriteDescriptorSets, nullptr);

		// Offscreen
		descriptorSets.offscreen = device.allocateDescriptorSets(allocInfo)[0];

		std::vector<vk::WriteDescriptorSet> offScreenWriteDescriptorSets =
		{
			// Binding 0 : Vertex shader uniform buffer
			vks::initializers::writeDescriptorSet(
				descriptorSets.offscreen,
				vk::DescriptorType::eUniformBuffer,
				0,
				&uniformBuffers.vsOffScreen.descriptor)
		};
		device.updateDescriptorSets(offScreenWriteDescriptorSets, nullptr);
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
				vk::CullModeFlagBits::eFront,
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

		// Solid rendering pipeline
		// Load shaders
		std::array<vk::PipelineShaderStageCreateInfo, 2> shaderStages;

		shaderStages[0] = loadShader(getAssetPath() + "shaders/offscreen/quad.vert.spv", vk::ShaderStageFlagBits::eVertex);
		shaderStages[1] = loadShader(getAssetPath() + "shaders/offscreen/quad.frag.spv", vk::ShaderStageFlagBits::eFragment);

		vk::GraphicsPipelineCreateInfo pipelineCreateInfo =
			vks::initializers::pipelineCreateInfo(
				pipelineLayouts.textured,
				renderPass);

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

		pipelines.debug = device.createGraphicsPipelines(pipelineCache, pipelineCreateInfo)[0];

		// Mirror
		shaderStages[0] = loadShader(getAssetPath() + "shaders/offscreen/mirror.vert.spv", vk::ShaderStageFlagBits::eVertex);
		shaderStages[1] = loadShader(getAssetPath() + "shaders/offscreen/mirror.frag.spv", vk::ShaderStageFlagBits::eFragment);
		rasterizationState.cullMode = vk::CullModeFlagBits::eNone;
		pipelines.mirror = device.createGraphicsPipelines(pipelineCache, pipelineCreateInfo)[0];

		// Flip culling
		rasterizationState.cullMode = vk::CullModeFlagBits::eBack;

		// Phong shading pipelines
		pipelineCreateInfo.layout = pipelineLayouts.shaded;
		// Scene
		shaderStages[0] = loadShader(getAssetPath() + "shaders/offscreen/phong.vert.spv", vk::ShaderStageFlagBits::eVertex);
		shaderStages[1] = loadShader(getAssetPath() + "shaders/offscreen/phong.frag.spv", vk::ShaderStageFlagBits::eFragment);
		pipelines.shaded = device.createGraphicsPipelines(pipelineCache, pipelineCreateInfo)[0];
		// Offscreen
		// Flip culling
		rasterizationState.cullMode = vk::CullModeFlagBits::eFront;
		pipelineCreateInfo.renderPass = offscreenPass.renderPass;
		pipelines.shadedOffscreen = device.createGraphicsPipelines(pipelineCache, pipelineCreateInfo)[0];

	}

	// Prepare and initialize uniform buffer containing shader uniforms
	void prepareUniformBuffers()
	{
		// Mesh vertex shader uniform buffer block
		vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eUniformBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			&uniformBuffers.vsShared,
			sizeof(uboShared));

		// Mirror plane vertex shader uniform buffer block
		vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eUniformBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			&uniformBuffers.vsMirror,
			sizeof(uboShared));

		// Offscreen vertex shader uniform buffer block 
		vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eUniformBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			&uniformBuffers.vsOffScreen,
			sizeof(uboShared));

		// Debug quad vertex shader uniform buffer block 
		vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eUniformBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			&uniformBuffers.vsDebugQuad,
			sizeof(uboShared));

		// Map persistent
		uniformBuffers.vsShared.map();
		uniformBuffers.vsMirror.map();
		uniformBuffers.vsOffScreen.map();
		uniformBuffers.vsDebugQuad.map();

		updateUniformBuffers();
		updateUniformBufferOffscreen();
	}

	void updateUniformBuffers()
	{
		// Mesh
		uboShared.projection = glm::perspective(glm::radians(60.0f), (float)width / (float)height, 0.1f, 256.0f);
		glm::mat4 viewMatrix = glm::translate(glm::mat4(), glm::vec3(0.0f, 0.0f, zoom));

		uboShared.model = viewMatrix * glm::translate(glm::mat4(), cameraPos);
		uboShared.model = glm::rotate(uboShared.model, glm::radians(rotation.x), glm::vec3(1.0f, 0.0f, 0.0f));
		uboShared.model = glm::rotate(uboShared.model, glm::radians(rotation.y + meshRot.y), glm::vec3(0.0f, 1.0f, 0.0f));
		uboShared.model = glm::rotate(uboShared.model, glm::radians(rotation.z), glm::vec3(0.0f, 0.0f, 1.0f));

		uboShared.model = glm::translate(uboShared.model, meshPos);

		memcpy(uniformBuffers.vsShared.mapped, &uboShared, sizeof(uboShared));

		// Mirror
		uboShared.model = viewMatrix * glm::translate(glm::mat4(), cameraPos);
		uboShared.model = glm::rotate(uboShared.model, glm::radians(rotation.x), glm::vec3(1.0f, 0.0f, 0.0f));
		uboShared.model = glm::rotate(uboShared.model, glm::radians(rotation.y), glm::vec3(0.0f, 1.0f, 0.0f));
		uboShared.model = glm::rotate(uboShared.model, glm::radians(rotation.z), glm::vec3(0.0f, 0.0f, 1.0f));

		memcpy(uniformBuffers.vsMirror.mapped, &uboShared, sizeof(uboShared));

		// Debug quad
		uboShared.projection = glm::ortho(4.0f, 0.0f, 0.0f, 4.0f*(float)height / (float)width, -1.0f, 1.0f);
		uboShared.model = glm::translate(glm::mat4(), glm::vec3(0.0f, 0.0f, 0.0f));

		memcpy(uniformBuffers.vsDebugQuad.mapped, &uboShared, sizeof(uboShared));
	}

	void updateUniformBufferOffscreen()
	{
		uboShared.projection = glm::perspective(glm::radians(60.0f), (float)width / (float)height, 0.1f, 256.0f);
		glm::mat4 viewMatrix = glm::translate(glm::mat4(), glm::vec3(0.0f, 0.0f, zoom));

		uboShared.model = viewMatrix * glm::translate(glm::mat4(), cameraPos);
		uboShared.model = glm::rotate(uboShared.model, glm::radians(rotation.x), glm::vec3(1.0f, 0.0f, 0.0f));
		uboShared.model = glm::rotate(uboShared.model, glm::radians(rotation.y + meshRot.y), glm::vec3(0.0f, 1.0f, 0.0f));
		uboShared.model = glm::rotate(uboShared.model, glm::radians(rotation.z), glm::vec3(0.0f, 0.0f, 1.0f));

		uboShared.model = glm::scale(uboShared.model, glm::vec3(1.0f, -1.0f, 1.0f));
		uboShared.model = glm::translate(uboShared.model, meshPos);

		memcpy(uniformBuffers.vsOffScreen.mapped, &uboShared, sizeof(uboShared));
	}

	void draw()
	{
		VulkanExampleBase::prepareFrame();

		// The scene render command buffer has to wait for the offscreen
		// rendering to be finished before we can use the framebuffer 
		// color image for sampling during final rendering
		// To ensure this we use a dedicated offscreen synchronization
		// semaphore that will be signaled when offscreen renderin
		// has been finished
		// This is necessary as an implementation may start both
		// command buffers at the same time, there is no guarantee
		// that command buffers will be executed in the order they
		// have been submitted by the application

		// Offscreen rendering

		// Wait for swap chain presentation to finish
		submitInfo.pWaitSemaphores = &semaphores.presentComplete;
		// Signal ready with offscreen semaphore
		submitInfo.pSignalSemaphores = &offscreenPass.semaphore;

		// Submit work
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &offscreenPass.commandBuffer;
		queue.submit(submitInfo, vk::Fence(nullptr));

		// Scene rendering

		// Wait for offscreen semaphore
		submitInfo.pWaitSemaphores = &offscreenPass.semaphore;
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
		generateQuad();
		prepareOffscreen();
		setupVertexDescriptions();
		prepareUniformBuffers();
		setupDescriptorSetLayout();
		preparePipelines();
		setupDescriptorPool();
		setupDescriptorSet();
		buildCommandBuffers();
		buildOffscreenCommandBuffer(); 
		prepared = true;
	}

	virtual void render()
	{
		if (!prepared)
			return;
		draw();
		if (!paused)
		{
			meshRot.y += frameTimer * 10.0f;
			updateUniformBuffers();
			updateUniformBufferOffscreen();
		}
	}

	virtual void viewChanged()
	{
		updateUniformBuffers();
		updateUniformBufferOffscreen();
	}

	virtual void keyPressed(uint32_t keyCode)
	{
		switch (keyCode)
		{
		case KEY_D:
		case GAMEPAD_BUTTON_A:
			toggleDebugDisplay();
			break;
		}
	}

	virtual void getOverlayText(VulkanTextOverlay *textOverlay)
	{
#if defined(__ANDROID__)
		textOverlay->addText("Press \"Button A\" to display offscreen target", 5.0f, 85.0f, VulkanTextOverlay::alignLeft);
#else
		textOverlay->addText("Press \"d\" to display offscreen target", 5.0f, 85.0f, VulkanTextOverlay::alignLeft);
#endif
	}

	void toggleDebugDisplay()
	{
		debugDisplay = !debugDisplay;
		reBuildCommandBuffers();
	}
};

VULKAN_EXAMPLE_MAIN()
