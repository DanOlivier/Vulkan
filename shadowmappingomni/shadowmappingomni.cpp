/*
* Vulkan Example - Omni directional shadows using a dynamic cube map
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

// Texture properties
#define TEX_DIM 1024
#define TEX_FILTER vk::Filter::eLinear

// Offscreen frame buffer properties
#define FB_DIM TEX_DIM
#define FB_COLOR_FORMAT vk::Format::eR32Sfloat 

class VulkanExample : public VulkanExampleBase
{
public:
	bool displayCubeMap = false;

	float zNear = 0.1f;
	float zFar = 1024.0f;

	struct {
		vk::PipelineVertexInputStateCreateInfo inputState;
		std::vector<vk::VertexInputBindingDescription> bindingDescriptions;
		std::vector<vk::VertexInputAttributeDescription> attributeDescriptions;
	} vertices;

	// Vertex layout for the models
	vks::VertexLayout vertexLayout = vks::VertexLayout({
		vks::VERTEX_COMPONENT_POSITION,
		vks::VERTEX_COMPONENT_UV,
		vks::VERTEX_COMPONENT_COLOR,
		vks::VERTEX_COMPONENT_NORMAL,
	});

	struct {
		vks::Model skybox;
		vks::Model scene;
	} models;

	struct {
		vks::Buffer scene;
		vks::Buffer offscreen;
	} uniformBuffers;

	struct {
		glm::mat4 projection;
		glm::mat4 model;
	} uboVSquad;

	glm::vec4 lightPos = glm::vec4(0.0f, -25.0f, 0.0f, 1.0); 

	struct UBO {
		glm::mat4 projection;
		glm::mat4 view;
		glm::mat4 model;
		glm::vec4 lightPos;
	};

	UBO uboVSscene, uboOffscreenVS;

	struct {
		vk::Pipeline scene;
		vk::Pipeline offscreen;
		vk::Pipeline cubeMap;
	} pipelines;

	struct {
		vk::PipelineLayout scene; 
		vk::PipelineLayout offscreen;
	} pipelineLayouts;

	struct {
		vk::DescriptorSet scene;
		vk::DescriptorSet offscreen;
	} descriptorSets;

	vk::DescriptorSetLayout descriptorSetLayout;

	vks::Texture shadowCubeMap;

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

	vk::Format fbDepthFormat;

	VulkanExample() : VulkanExampleBase(ENABLE_VALIDATION)
	{
		zoom = -175.0f;
		zoomSpeed = 10.0f;
		timerSpeed *= 0.25f;
		rotation = { -20.5f, -673.0f, 0.0f };
		enableTextOverlay = true;
		title = "Vulkan Example - Point light shadows";
	}

	~VulkanExample()
	{
		// Clean up used Vulkan resources 
		// Note : Inherited destructor cleans up resources stored in base class

		// Cube map
		device.destroyImageView(shadowCubeMap.view);
		device.destroyImage(shadowCubeMap.image);
		device.destroySampler(shadowCubeMap.sampler);
		device.freeMemory(shadowCubeMap.deviceMemory);

		// Frame buffer

		// Color attachment
		device.destroyImageView(offscreenPass.color.view);
		device.destroyImage(offscreenPass.color.image);
		device.freeMemory(offscreenPass.color.mem);

		// Depth attachment
		device.destroyImageView(offscreenPass.depth.view);
		device.destroyImage(offscreenPass.depth.image);
		device.freeMemory(offscreenPass.depth.mem);

		device.destroyFramebuffer(offscreenPass.frameBuffer);

		device.destroyRenderPass(offscreenPass.renderPass);

		// Pipelibes
		device.destroyPipeline(pipelines.scene);
		device.destroyPipeline(pipelines.offscreen);
		device.destroyPipeline(pipelines.cubeMap);

		device.destroyPipelineLayout(pipelineLayouts.scene);
		device.destroyPipelineLayout(pipelineLayouts.offscreen);

		device.destroyDescriptorSetLayout(descriptorSetLayout);

		// Meshes
		models.scene.destroy();
		models.skybox.destroy();

		// Uniform buffers
		uniformBuffers.offscreen.destroy();
		uniformBuffers.scene.destroy();

		device.freeCommandBuffers(cmdPool, offscreenPass.commandBuffer);
		device.destroySemaphore(offscreenPass.semaphore);
	}

	void prepareCubeMap()
	{
		shadowCubeMap.width = TEX_DIM;
		shadowCubeMap.height = TEX_DIM;
		
		// 32 bit float format for higher precision
		vk::Format format = vk::Format::eR32Sfloat;

		// Cube map image description
		vk::ImageCreateInfo imageCreateInfo = vks::initializers::imageCreateInfo();
		imageCreateInfo.imageType = vk::ImageType::e2D;
		imageCreateInfo.format = format;
		imageCreateInfo.extent = vk::Extent3D{ shadowCubeMap.width, shadowCubeMap.height, 1 };
		imageCreateInfo.mipLevels = 1;
		imageCreateInfo.arrayLayers = 6;
		imageCreateInfo.samples = vk::SampleCountFlagBits::e1;
		imageCreateInfo.tiling = vk::ImageTiling::eOptimal;
		imageCreateInfo.usage = vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled;
		imageCreateInfo.sharingMode = vk::SharingMode::eExclusive;
		imageCreateInfo.initialLayout = vk::ImageLayout::eUndefined;
		imageCreateInfo.flags = vk::ImageCreateFlagBits::eCubeCompatible;

		vk::MemoryAllocateInfo memAllocInfo = vks::initializers::memoryAllocateInfo();
		vk::MemoryRequirements memReqs;

		vk::CommandBuffer layoutCmd = VulkanExampleBase::createCommandBuffer(vk::CommandBufferLevel::ePrimary, true);

		// Create cube map image
		shadowCubeMap.image = device.createImage(imageCreateInfo);

		memReqs = device.getImageMemoryRequirements(shadowCubeMap.image);

		memAllocInfo.allocationSize = memReqs.size;
		memAllocInfo.memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);
		shadowCubeMap.deviceMemory = device.allocateMemory(memAllocInfo);
		device.bindImageMemory(shadowCubeMap.image, shadowCubeMap.deviceMemory, 0);

		// Image barrier for optimal image (target)
		vk::ImageSubresourceRange subresourceRange;
		subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
		subresourceRange.baseMipLevel = 0;
		subresourceRange.levelCount = 1;
		subresourceRange.layerCount = 6;
		vks::tools::setImageLayout(
			layoutCmd,
			shadowCubeMap.image,
			vk::ImageLayout::eUndefined,
			vk::ImageLayout::eShaderReadOnlyOptimal,
			subresourceRange);

		VulkanExampleBase::flushCommandBuffer(layoutCmd, queue, true);

		// Create sampler
		vk::SamplerCreateInfo sampler = vks::initializers::samplerCreateInfo();
		sampler.magFilter = TEX_FILTER;
		sampler.minFilter = TEX_FILTER;
		sampler.mipmapMode = vk::SamplerMipmapMode::eLinear;
		sampler.addressModeU = vk::SamplerAddressMode::eClampToBorder;
		sampler.addressModeV = sampler.addressModeU;
		sampler.addressModeW = sampler.addressModeU;
		sampler.mipLodBias = 0.0f;
		sampler.maxAnisotropy = 1.0f;
		sampler.compareOp = vk::CompareOp::eNever;
		sampler.minLod = 0.0f;
		sampler.maxLod = 1.0f;
		sampler.borderColor = vk::BorderColor::eFloatOpaqueWhite;
		shadowCubeMap.sampler = device.createSampler(sampler);

		// Create image view
		vk::ImageViewCreateInfo view = vks::initializers::imageViewCreateInfo();
		view.image;
		view.viewType = vk::ImageViewType::eCube;
		view.format = format;
		view.components = { vk::ComponentSwizzle::eR };
		view.subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 };
		view.subresourceRange.layerCount = 6;
		view.image = shadowCubeMap.image;
		shadowCubeMap.view = device.createImageView(view);
	}

	// Prepare a new framebuffer for offscreen rendering
	// The contents of this framebuffer are then
	// copied to the different cube map faces
	void prepareOffscreenFramebuffer()
	{
		offscreenPass.width = FB_DIM;
		offscreenPass.height = FB_DIM;

		vk::Format fbColorFormat = FB_COLOR_FORMAT;

		// Color attachment
		vk::ImageCreateInfo imageCreateInfo = vks::initializers::imageCreateInfo();
		imageCreateInfo.imageType = vk::ImageType::e2D;
		imageCreateInfo.format = fbColorFormat;
		imageCreateInfo.extent.width = offscreenPass.width;
		imageCreateInfo.extent.height = offscreenPass.height;
		imageCreateInfo.extent.depth = 1;
		imageCreateInfo.mipLevels = 1;
		imageCreateInfo.arrayLayers = 1;
		imageCreateInfo.samples = vk::SampleCountFlagBits::e1;
		imageCreateInfo.tiling = vk::ImageTiling::eOptimal;
		// Image of the framebuffer is blit source
		imageCreateInfo.initialLayout = vk::ImageLayout::eUndefined;
		imageCreateInfo.usage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferSrc;
		imageCreateInfo.sharingMode = vk::SharingMode::eExclusive;

		vk::MemoryAllocateInfo memAlloc = vks::initializers::memoryAllocateInfo();

		vk::ImageViewCreateInfo colorImageView = vks::initializers::imageViewCreateInfo();
		colorImageView.viewType = vk::ImageViewType::e2D;
		colorImageView.format = fbColorFormat;
		colorImageView.flags = 0;
		//colorImageView.subresourceRange = {};
		colorImageView.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
		colorImageView.subresourceRange.baseMipLevel = 0;
		colorImageView.subresourceRange.levelCount = 1;
		colorImageView.subresourceRange.baseArrayLayer = 0;
		colorImageView.subresourceRange.layerCount = 1;

		vk::MemoryRequirements memReqs;

		offscreenPass.color.image = device.createImage(imageCreateInfo);
		memReqs = device.getImageMemoryRequirements(offscreenPass.color.image);
		memAlloc.allocationSize = memReqs.size;
		memAlloc.memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);
		offscreenPass.color.mem = device.allocateMemory(memAlloc);
		device.bindImageMemory(offscreenPass.color.image, offscreenPass.color.mem, 0);

		vk::CommandBuffer layoutCmd = VulkanExampleBase::createCommandBuffer(vk::CommandBufferLevel::ePrimary, true);

		vks::tools::setImageLayout(
			layoutCmd,
			offscreenPass.color.image, 
			vk::ImageAspectFlagBits::eColor, 
			vk::ImageLayout::eUndefined, 
			vk::ImageLayout::eColorAttachmentOptimal);

		colorImageView.image = offscreenPass.color.image;
		offscreenPass.color.view = device.createImageView(colorImageView);

		// Depth stencil attachment
		imageCreateInfo.format = fbDepthFormat;
		imageCreateInfo.usage = vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eTransferSrc;

		vk::ImageViewCreateInfo depthStencilView = vks::initializers::imageViewCreateInfo();
		depthStencilView.viewType = vk::ImageViewType::e2D;
		depthStencilView.format = fbDepthFormat;
		depthStencilView.flags = 0;
		//depthStencilView.subresourceRange = {};
		depthStencilView.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil;
		depthStencilView.subresourceRange.baseMipLevel = 0;
		depthStencilView.subresourceRange.levelCount = 1;
		depthStencilView.subresourceRange.baseArrayLayer = 0;
		depthStencilView.subresourceRange.layerCount = 1;

		offscreenPass.depth.image = device.createImage(imageCreateInfo);
		memReqs = device.getImageMemoryRequirements(offscreenPass.depth.image);
		memAlloc.allocationSize = memReqs.size;
		memAlloc.memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);
		offscreenPass.depth.mem = device.allocateMemory(memAlloc);
		device.bindImageMemory(offscreenPass.depth.image, offscreenPass.depth.mem, 0);

		vks::tools::setImageLayout(
			layoutCmd,
			offscreenPass.depth.image, 
			vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil,
			vk::ImageLayout::eUndefined,
			vk::ImageLayout::eDepthStencilAttachmentOptimal);

		VulkanExampleBase::flushCommandBuffer(layoutCmd, queue, true);

		depthStencilView.image = offscreenPass.depth.image;
		offscreenPass.depth.view = device.createImageView(depthStencilView);

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
	}

	// Updates a single cube map face
	// Renders the scene with face's view and does 
	// a copy from framebuffer to cube face
	// Uses push constants for quick update of
	// view matrix for the current cube map face
	void updateCubeFace(uint32_t faceIndex)
	{
		vk::ClearValue clearValues[2];
		clearValues[0].color = vk::ClearColorValue{ std::array<float, 4>{ 0.0f, 0.0f, 0.0f, 1.0f } };
		clearValues[1].depthStencil = vk::ClearDepthStencilValue{ 1.0f, 0 };

		vk::RenderPassBeginInfo renderPassBeginInfo = vks::initializers::renderPassBeginInfo();
		// Reuse render pass from example pass
		renderPassBeginInfo.renderPass = offscreenPass.renderPass;
		renderPassBeginInfo.framebuffer = offscreenPass.frameBuffer;
		renderPassBeginInfo.renderArea.extent.width = offscreenPass.width;
		renderPassBeginInfo.renderArea.extent.height = offscreenPass.height;
		renderPassBeginInfo.clearValueCount = 2;
		renderPassBeginInfo.pClearValues = clearValues;

		// Update view matrix via push constant

		glm::mat4 viewMatrix = glm::mat4();
		switch (faceIndex)
		{
		case 0: // POSITIVE_X
			viewMatrix = glm::rotate(viewMatrix, glm::radians(90.0f), glm::vec3(0.0f, 1.0f, 0.0f));
			viewMatrix = glm::rotate(viewMatrix, glm::radians(180.0f), glm::vec3(1.0f, 0.0f, 0.0f));
			break;
		case 1:	// NEGATIVE_X
			viewMatrix = glm::rotate(viewMatrix, glm::radians(-90.0f), glm::vec3(0.0f, 1.0f, 0.0f));
			viewMatrix = glm::rotate(viewMatrix, glm::radians(180.0f), glm::vec3(1.0f, 0.0f, 0.0f));
			break;
		case 2:	// POSITIVE_Y
			viewMatrix = glm::rotate(viewMatrix, glm::radians(-90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
			break;
		case 3:	// NEGATIVE_Y
			viewMatrix = glm::rotate(viewMatrix, glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
			break;
		case 4:	// POSITIVE_Z
			viewMatrix = glm::rotate(viewMatrix, glm::radians(180.0f), glm::vec3(1.0f, 0.0f, 0.0f));
			break;
		case 5:	// NEGATIVE_Z
			viewMatrix = glm::rotate(viewMatrix, glm::radians(180.0f), glm::vec3(0.0f, 0.0f, 1.0f));
			break;
		}

		// Render scene from cube face's point of view
		offscreenPass.commandBuffer.beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);

		// Update shader push constant block
		// Contains current face view matrix
		offscreenPass.commandBuffer.pushConstants(
			pipelineLayouts.offscreen,
			vk::ShaderStageFlagBits::eVertex,
			0,
			viewMatrix);

		offscreenPass.commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.offscreen);
		offscreenPass.commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayouts.offscreen, 0, descriptorSets.offscreen, nullptr);

		std::vector<vk::DeviceSize> offsets = { 0 };
		offscreenPass.commandBuffer.bindVertexBuffers(VERTEX_BUFFER_BIND_ID, models.scene.vertices.buffer, offsets);
		offscreenPass.commandBuffer.bindIndexBuffer(models.scene.indices.buffer, 0, vk::IndexType::eUint32);
		offscreenPass.commandBuffer.drawIndexed(models.scene.indexCount, 1, 0, 0, 0);

		offscreenPass.commandBuffer.endRenderPass();
		// Make sure color writes to the framebuffer are finished before using it as transfer source
		vks::tools::setImageLayout(
			offscreenPass.commandBuffer,
			offscreenPass.color.image,
			vk::ImageAspectFlagBits::eColor,
			vk::ImageLayout::eColorAttachmentOptimal,
			vk::ImageLayout::eTransferSrcOptimal);

		vk::ImageSubresourceRange cubeFaceSubresourceRange = {};
		cubeFaceSubresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
		cubeFaceSubresourceRange.baseMipLevel = 0;
		cubeFaceSubresourceRange.levelCount = 1;
		cubeFaceSubresourceRange.baseArrayLayer = faceIndex;
		cubeFaceSubresourceRange.layerCount = 1;

		// Change image layout of one cubemap face to transfer destination
		vks::tools::setImageLayout(
			offscreenPass.commandBuffer,
			shadowCubeMap.image,
			vk::ImageLayout::eShaderReadOnlyOptimal,
			vk::ImageLayout::eTransferDstOptimal,
			cubeFaceSubresourceRange);

		// Copy region for transfer from framebuffer to cube face
		vk::ImageCopy copyRegion = {};

		copyRegion.srcSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
		copyRegion.srcSubresource.baseArrayLayer = 0;
		copyRegion.srcSubresource.mipLevel = 0;
		copyRegion.srcSubresource.layerCount = 1;
		copyRegion.srcOffset = { 0, 0, 0 };

		copyRegion.dstSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
		copyRegion.dstSubresource.baseArrayLayer = faceIndex;
		copyRegion.dstSubresource.mipLevel = 0;
		copyRegion.dstSubresource.layerCount = 1;
		copyRegion.dstOffset = { 0, 0, 0 };

		copyRegion.extent.width = shadowCubeMap.width;
		copyRegion.extent.height = shadowCubeMap.height;
		copyRegion.extent.depth = 1;

		// Put image copy into command buffer
		offscreenPass.commandBuffer.copyImage(
			offscreenPass.color.image,
			vk::ImageLayout::eTransferSrcOptimal,
			shadowCubeMap.image,
			vk::ImageLayout::eTransferDstOptimal,
			copyRegion);

		// Transform framebuffer color attachment back 
		vks::tools::setImageLayout(
			offscreenPass.commandBuffer,
			offscreenPass.color.image,
			vk::ImageAspectFlagBits::eColor,
			vk::ImageLayout::eTransferSrcOptimal,
			vk::ImageLayout::eColorAttachmentOptimal);

		// Change image layout of copied face to shader read
		vks::tools::setImageLayout(
			offscreenPass.commandBuffer,
			shadowCubeMap.image,
			vk::ImageLayout::eTransferDstOptimal,
			vk::ImageLayout::eShaderReadOnlyOptimal,
			cubeFaceSubresourceRange);
	}

	// Command buffer for rendering and copying all cube map faces
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

		offscreenPass.commandBuffer.begin(cmdBufInfo);

		vk::Viewport viewport = vks::initializers::viewport((float)offscreenPass.width, (float)offscreenPass.height, 0.0f, 1.0f);
		offscreenPass.commandBuffer.setViewport(0, viewport);

		vk::Rect2D scissor = vks::initializers::rect2D(offscreenPass.width, offscreenPass.height,	0, 0);
		offscreenPass.commandBuffer.setScissor(0, scissor);

		for (uint32_t face = 0; face < 6; ++face)
		{
			updateCubeFace(face);
		}

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
			renderPassBeginInfo.framebuffer = frameBuffers[i];

			drawCmdBuffers[i].begin(cmdBufInfo);

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
			drawCmdBuffers[i].setScissor(0, scissor);

			std::vector<vk::DeviceSize> offsets = { 0 };

			drawCmdBuffers[i].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayouts.scene, 0, descriptorSets.scene, nullptr);

			if (displayCubeMap)
			{
				drawCmdBuffers[i].bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.cubeMap);
				drawCmdBuffers[i].bindVertexBuffers(VERTEX_BUFFER_BIND_ID, models.skybox.vertices.buffer, offsets);
				drawCmdBuffers[i].bindIndexBuffer(models.skybox.indices.buffer, 0, vk::IndexType::eUint32);
				drawCmdBuffers[i].drawIndexed(models.skybox.indexCount, 1, 0, 0, 0);
			}
			else
			{
				drawCmdBuffers[i].bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.scene);
				drawCmdBuffers[i].bindVertexBuffers(VERTEX_BUFFER_BIND_ID, models.scene.vertices.buffer, offsets);
				drawCmdBuffers[i].bindIndexBuffer(models.scene.indices.buffer, 0, vk::IndexType::eUint32);
				drawCmdBuffers[i].drawIndexed(models.scene.indexCount, 1, 0, 0, 0);
			}

			drawCmdBuffers[i].endRenderPass();

			drawCmdBuffers[i].end();
		}
	}	

	void loadAssets()
	{
		models.skybox.loadFromFile(getAssetPath() + "models/cube.obj", vertexLayout, 2.0f, vulkanDevice, queue);
		models.scene.loadFromFile(getAssetPath() + "models/shadowscene_fire.dae", vertexLayout, 2.0f, vulkanDevice, queue);
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
		// Example uses three ubos and two image samplers
		std::vector<vk::DescriptorPoolSize> poolSizes =
		{
			vks::initializers::descriptorPoolSize(vk::DescriptorType::eUniformBuffer, 3),
			vks::initializers::descriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, 2)
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
		// Shared pipeline layout
		std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings =
		{
			// Binding 0 : Vertex shader uniform buffer
			vks::initializers::descriptorSetLayoutBinding(
				vk::DescriptorType::eUniformBuffer,
				vk::ShaderStageFlagBits::eVertex,
				0),
			// Binding 1 : Fragment shader image sampler (cube map)
			vks::initializers::descriptorSetLayoutBinding(
				vk::DescriptorType::eCombinedImageSampler,
				vk::ShaderStageFlagBits::eFragment,
				1)
		};

		vk::DescriptorSetLayoutCreateInfo descriptorLayout =
			vks::initializers::descriptorSetLayoutCreateInfo(
				setLayoutBindings.data(),
				setLayoutBindings.size());

		descriptorSetLayout = device.createDescriptorSetLayout(descriptorLayout);

		// 3D scene pipeline layout
		vk::PipelineLayoutCreateInfo pPipelineLayoutCreateInfo =
			vks::initializers::pipelineLayoutCreateInfo(
				&descriptorSetLayout,
				1);

		pipelineLayouts.scene = device.createPipelineLayout(pPipelineLayoutCreateInfo);

		// Offscreen pipeline layout
		// Push constants for cube map face view matrices
		vk::PushConstantRange pushConstantRange =
			vks::initializers::pushConstantRange(
				vk::ShaderStageFlagBits::eVertex,
				sizeof(glm::mat4),
				0);

		// Push constant ranges are part of the pipeline layout
		pPipelineLayoutCreateInfo.pushConstantRangeCount = 1;
		pPipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRange;

		pipelineLayouts.offscreen = device.createPipelineLayout(pPipelineLayoutCreateInfo);
	}

	void setupDescriptorSets()
	{
		vk::DescriptorSetAllocateInfo allocInfo =
			vks::initializers::descriptorSetAllocateInfo(
				descriptorPool,
				&descriptorSetLayout,
				1);

		// 3D scene
		descriptorSets.scene = device.allocateDescriptorSets(allocInfo)[0];

		// Image descriptor for the cube map 
		vk::DescriptorImageInfo texDescriptor =
			vks::initializers::descriptorImageInfo(
				shadowCubeMap.sampler,
				shadowCubeMap.view,
				vk::ImageLayout::eShaderReadOnlyOptimal);

		std::vector<vk::WriteDescriptorSet> sceneDescriptorSets =
		{
			// Binding 0 : Vertex shader uniform buffer
			vks::initializers::writeDescriptorSet(
			descriptorSets.scene,
				vk::DescriptorType::eUniformBuffer,
				0,
				&uniformBuffers.scene.descriptor),
			// Binding 1 : Fragment shader shadow sampler
			vks::initializers::writeDescriptorSet(
				descriptorSets.scene,
				vk::DescriptorType::eCombinedImageSampler,
				1,
				&texDescriptor)
		};
		device.updateDescriptorSets(sceneDescriptorSets, nullptr);

		// Offscreen
		descriptorSets.offscreen = device.allocateDescriptorSets(allocInfo)[0];

		std::vector<vk::WriteDescriptorSet> offScreenWriteDescriptorSets =
		{
			// Binding 0 : Vertex shader uniform buffer
			vks::initializers::writeDescriptorSet(
				descriptorSets.offscreen,
				vk::DescriptorType::eUniformBuffer,
				0,
				&uniformBuffers.offscreen.descriptor),
		};
		device.updateDescriptorSets(offScreenWriteDescriptorSets, nullptr);
	}

	// Set up a separate render pass for the offscreen frame buffer
	// This is necessary as the offscreen frame buffer attachments
	// use formats different to the ones from the visible frame buffer
	// and at least the depth one may not be compatible
	void prepareOffscreenRenderpass()
	{
		vk::AttachmentDescription osAttachments[2] = {};

		// Find a suitable depth format
		vk::Bool32 validDepthFormat = vks::tools::getSupportedDepthFormat(physicalDevice, &fbDepthFormat);
		assert(validDepthFormat);

		osAttachments[0].format = FB_COLOR_FORMAT;
		osAttachments[0].samples = vk::SampleCountFlagBits::e1;
		osAttachments[0].loadOp = vk::AttachmentLoadOp::eClear;
		osAttachments[0].storeOp = vk::AttachmentStoreOp::eStore;
		osAttachments[0].stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
		osAttachments[0].stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
		osAttachments[0].initialLayout = vk::ImageLayout::eColorAttachmentOptimal;
		osAttachments[0].finalLayout = vk::ImageLayout::eColorAttachmentOptimal;

		// Depth attachment
		osAttachments[1].format = fbDepthFormat;
		osAttachments[1].samples = vk::SampleCountFlagBits::e1;
		osAttachments[1].loadOp = vk::AttachmentLoadOp::eClear;
		osAttachments[1].storeOp = vk::AttachmentStoreOp::eStore;
		osAttachments[1].stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
		osAttachments[1].stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
		osAttachments[1].initialLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal;
		osAttachments[1].finalLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal;

		vk::AttachmentReference colorReference = {};
		colorReference.attachment = 0;
		colorReference.layout = vk::ImageLayout::eColorAttachmentOptimal;

		vk::AttachmentReference depthReference = {};
		depthReference.attachment = 1;
		depthReference.layout = vk::ImageLayout::eDepthStencilAttachmentOptimal;

		vk::SubpassDescription subpass = {};
		subpass.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
		subpass.colorAttachmentCount = 1;
		subpass.pColorAttachments = &colorReference;
		subpass.pDepthStencilAttachment = &depthReference;

		vk::RenderPassCreateInfo renderPassCreateInfo = vks::initializers::renderPassCreateInfo();
		renderPassCreateInfo.attachmentCount = 2;
		renderPassCreateInfo.pAttachments = osAttachments;
		renderPassCreateInfo.subpassCount = 1;
		renderPassCreateInfo.pSubpasses = &subpass;

		offscreenPass.renderPass = device.createRenderPass(renderPassCreateInfo);
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

		// 3D scene pipeline
		// Load shaders
		std::array<vk::PipelineShaderStageCreateInfo, 2> shaderStages;

		shaderStages[0] = loadShader(getAssetPath() + "shaders/shadowmapomni/scene.vert.spv", vk::ShaderStageFlagBits::eVertex);
		shaderStages[1] = loadShader(getAssetPath() + "shaders/shadowmapomni/scene.frag.spv", vk::ShaderStageFlagBits::eFragment);

		vk::GraphicsPipelineCreateInfo pipelineCreateInfo =
			vks::initializers::pipelineCreateInfo(
				pipelineLayouts.scene,
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

		pipelines.scene = device.createGraphicsPipelines(pipelineCache, pipelineCreateInfo)[0];

		// Cube map display pipeline
		shaderStages[0] = loadShader(getAssetPath() + "shaders/shadowmapomni/cubemapdisplay.vert.spv", vk::ShaderStageFlagBits::eVertex);
		shaderStages[1] = loadShader(getAssetPath() + "shaders/shadowmapomni/cubemapdisplay.frag.spv", vk::ShaderStageFlagBits::eFragment);
		rasterizationState.cullMode = vk::CullModeFlagBits::eFront;
		pipelines.cubeMap = device.createGraphicsPipelines(pipelineCache, pipelineCreateInfo)[0];

		// Offscreen pipeline
		shaderStages[0] = loadShader(getAssetPath() + "shaders/shadowmapomni/offscreen.vert.spv", vk::ShaderStageFlagBits::eVertex);
		shaderStages[1] = loadShader(getAssetPath() + "shaders/shadowmapomni/offscreen.frag.spv", vk::ShaderStageFlagBits::eFragment);
		rasterizationState.cullMode = vk::CullModeFlagBits::eBack;
		pipelineCreateInfo.layout = pipelineLayouts.offscreen;
		pipelineCreateInfo.renderPass = offscreenPass.renderPass;
		pipelines.offscreen = device.createGraphicsPipelines(pipelineCache, pipelineCreateInfo)[0];
	}

	// Prepare and initialize uniform buffer containing shader uniforms
	void prepareUniformBuffers()
	{
		// Offscreen vertex shader uniform buffer 
		vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eUniformBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			&uniformBuffers.offscreen,
			sizeof(uboOffscreenVS));

		// Scene vertex shader uniform buffer
		vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eUniformBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			&uniformBuffers.scene,
			sizeof(uboVSscene));

		// Map persistent
		uniformBuffers.offscreen.map();
		uniformBuffers.scene.map();

		updateUniformBufferOffscreen();
		updateUniformBuffers();
	}

	void updateUniformBuffers()
	{
		uboVSscene.projection = glm::perspective(glm::radians(45.0f), (float)width / (float)height, zNear, zFar);
		uboVSscene.view = glm::translate(glm::mat4(), glm::vec3(0.0f, 0.0f, displayCubeMap ? 0.0f : zoom));

		uboVSscene.model = glm::mat4();
		uboVSscene.model = glm::rotate(uboVSscene.model, glm::radians(rotation.x), glm::vec3(1.0f, 0.0f, 0.0f));
		uboVSscene.model = glm::rotate(uboVSscene.model, glm::radians(rotation.y), glm::vec3(0.0f, 1.0f, 0.0f));
		uboVSscene.model = glm::rotate(uboVSscene.model, glm::radians(rotation.z), glm::vec3(0.0f, 0.0f, 1.0f));

		uboVSscene.lightPos = lightPos;

		memcpy(uniformBuffers.scene.mapped, &uboVSscene, sizeof(uboVSscene));
	}

	void updateUniformBufferOffscreen()
	{
		lightPos.x = sin(glm::radians(timer * 360.0f)) * 1.0f;
		lightPos.z = cos(glm::radians(timer * 360.0f)) * 1.0f;

		uboOffscreenVS.projection = glm::perspective((float)(M_PI / 2.0), 1.0f, zNear, zFar);

		uboOffscreenVS.view = glm::mat4();
		uboOffscreenVS.model = glm::translate(glm::mat4(), glm::vec3(-lightPos.x, -lightPos.y, -lightPos.z));

		uboOffscreenVS.lightPos = lightPos;

		memcpy(uniformBuffers.offscreen.mapped, &uboOffscreenVS, sizeof(uboOffscreenVS));
	}

	void draw()
	{
		VulkanExampleBase::prepareFrame();

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
		setupVertexDescriptions();
		prepareUniformBuffers();
		prepareCubeMap();
		setupDescriptorSetLayout();
		prepareOffscreenRenderpass();
		preparePipelines();
		setupDescriptorPool();
		setupDescriptorSets();
		prepareOffscreenFramebuffer();
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
			updateUniformBufferOffscreen();
			updateUniformBuffers();
		}
	}

	virtual void viewChanged()
	{
		updateUniformBufferOffscreen();
		updateUniformBuffers();
	}

	virtual void keyPressed(uint32_t keyCode)
	{
		switch (keyCode)
		{
		case KEY_D:
		case GAMEPAD_BUTTON_A:
			toggleCubeMapDisplay();
			break;
		}
	}

	virtual void getOverlayText(VulkanTextOverlay *textOverlay)
	{
#if defined(__ANDROID__)
		textOverlay->addText("Press \"Button A\" to display depth cubemap", 5.0f, 85.0f, VulkanTextOverlay::alignLeft);
#else
		textOverlay->addText("Press \"d\" to display depth cubemap", 5.0f, 85.0f, VulkanTextOverlay::alignLeft);
#endif
	}

	void toggleCubeMapDisplay()
	{
		displayCubeMap = !displayCubeMap;
		reBuildCommandBuffers();
	}
};

VULKAN_EXAMPLE_MAIN()
