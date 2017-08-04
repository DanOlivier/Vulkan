/*
* Vulkan Example - Implements a separable two-pass fullscreen blur (also known as bloom)
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
#define FB_DIM 256
#define FB_COLOR_FORMAT vk::Format::eR8G8B8A8Unorm

class VulkanExample : public VulkanExampleBase
{
public:
	bool bloom = true;

	struct {
		vks::TextureCubeMap cubemap;
	} textures;

	// Vertex layout for the models
	vks::VertexLayout vertexLayout = vks::VertexLayout({
		vks::VERTEX_COMPONENT_POSITION,
		vks::VERTEX_COMPONENT_UV,
		vks::VERTEX_COMPONENT_COLOR,
		vks::VERTEX_COMPONENT_NORMAL,
	});

	struct {
		vks::Model ufo;
		vks::Model ufoGlow;
		vks::Model skyBox;
	} models;

	struct {
		vk::PipelineVertexInputStateCreateInfo inputState;
		std::vector<vk::VertexInputBindingDescription> bindingDescriptions;
		std::vector<vk::VertexInputAttributeDescription> attributeDescriptions;
	} vertices;

	struct {
		vks::Buffer scene;
		vks::Buffer skyBox;
		vks::Buffer blurParams;
	} uniformBuffers;

	struct UBO {
		glm::mat4 projection;
		glm::mat4 view;
		glm::mat4 model;
	};

	struct UBOBlurParams {
		float blurScale = 1.0f;
		float blurStrength = 1.5f;
	};

	struct {
		UBO scene, skyBox;
		UBOBlurParams blurParams;
	} ubos;

	struct {
		vk::Pipeline blurVert;
		vk::Pipeline blurHorz;
		vk::Pipeline glowPass;
		vk::Pipeline phongPass;
		vk::Pipeline skyBox;
	} pipelines;

	struct {
		vk::PipelineLayout blur;
		vk::PipelineLayout scene;
	} pipelineLayouts; 

	struct {
		vk::DescriptorSet blurVert;
		vk::DescriptorSet blurHorz;
		vk::DescriptorSet scene;
		vk::DescriptorSet skyBox;
	} descriptorSets;

	struct {
		vk::DescriptorSetLayout blur;
		vk::DescriptorSetLayout scene;
	} descriptorSetLayouts;

	// Framebuffer for offscreen rendering
	struct FrameBufferAttachment {
		vk::Image image;
		vk::DeviceMemory mem;
		vk::ImageView view;
	};
	struct FrameBuffer {
		vk::Framebuffer framebuffer;
		FrameBufferAttachment color, depth;
		vk::DescriptorImageInfo descriptor;
	};
	struct OffscreenPass {
		int32_t width, height;
		vk::RenderPass renderPass;
		vk::Sampler sampler;
		vk::CommandBuffer commandBuffer = VK_NULL_HANDLE;
		// Semaphore used to synchronize between offscreen and final scene rendering
		vk::Semaphore semaphore = VK_NULL_HANDLE;
		std::array<FrameBuffer, 2> framebuffers;
	} offscreenPass;

	VulkanExample() : VulkanExampleBase(ENABLE_VALIDATION)
	{
		title = "Vulkan Example - Bloom";
		timerSpeed *= 0.5f;
		enableTextOverlay = true;
		camera.type = Camera::CameraType::lookat;
		camera.setPosition(glm::vec3(0.0f, 0.0f, -10.25f));
		camera.setRotation(glm::vec3(7.5f, -343.0f, 0.0f));
		camera.setPerspective(45.0f, (float)width / (float)height, 0.1f, 256.0f);
	}

	~VulkanExample()
	{
		// Clean up used Vulkan resources 
		// Note : Inherited destructor cleans up resources stored in base class

		vkDestroySampler(device, offscreenPass.sampler, nullptr);

		// Frame buffer
		for (auto& framebuffer : offscreenPass.framebuffers)
		{
			// Attachments
			vkDestroyImageView(device, framebuffer.color.view, nullptr);
			vkDestroyImage(device, framebuffer.color.image, nullptr);
			vkFreeMemory(device, framebuffer.color.mem, nullptr);
			vkDestroyImageView(device, framebuffer.depth.view, nullptr);
			vkDestroyImage(device, framebuffer.depth.image, nullptr);
			vkFreeMemory(device, framebuffer.depth.mem, nullptr);

			vkDestroyFramebuffer(device, framebuffer.framebuffer, nullptr);
		}
		vkDestroyRenderPass(device, offscreenPass.renderPass, nullptr);
		vkFreeCommandBuffers(device, cmdPool, 1, &offscreenPass.commandBuffer);
		vkDestroySemaphore(device, offscreenPass.semaphore, nullptr);

		vkDestroyPipeline(device, pipelines.blurHorz, nullptr);
		vkDestroyPipeline(device, pipelines.blurVert, nullptr);
		vkDestroyPipeline(device, pipelines.phongPass, nullptr);
		vkDestroyPipeline(device, pipelines.glowPass, nullptr);
		vkDestroyPipeline(device, pipelines.skyBox, nullptr);

		vkDestroyPipelineLayout(device, pipelineLayouts.blur , nullptr);
		vkDestroyPipelineLayout(device, pipelineLayouts.scene, nullptr);

		vkDestroyDescriptorSetLayout(device, descriptorSetLayouts.blur, nullptr);
		vkDestroyDescriptorSetLayout(device, descriptorSetLayouts.scene, nullptr);

		// Models
		models.ufo.destroy();
		models.ufoGlow.destroy();
		models.skyBox.destroy();

		// Uniform buffers
		uniformBuffers.scene.destroy();
		uniformBuffers.skyBox.destroy();
		uniformBuffers.blurParams.destroy();

		textures.cubemap.destroy();
	}

	// Setup the offscreen framebuffer for rendering the mirrored scene
	// The color attachment of this framebuffer will then be sampled from
	void prepareOffscreenFramebuffer(FrameBuffer *frameBuf, vk::Format colorFormat, vk::Format depthFormat)
	{
		// Color attachment
		vk::ImageCreateInfo image = vks::initializers::imageCreateInfo();
		image.imageType = vk::ImageType::e2D;
		image.format = colorFormat;
		image.extent.width = FB_DIM;
		image.extent.height = FB_DIM;
		image.extent.depth = 1;
		image.mipLevels = 1;
		image.arrayLayers = 1;
		image.samples = vk::SampleCountFlagBits::e1;
		image.tiling = vk::ImageTiling::eOptimal;
		// We will sample directly from the color attachment
		image.usage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled;

		vk::MemoryAllocateInfo memAlloc = vks::initializers::memoryAllocateInfo();
		vk::MemoryRequirements memReqs;

		vk::ImageViewCreateInfo colorImageView = vks::initializers::imageViewCreateInfo();
		colorImageView.viewType = vk::ImageViewType::e2D;
		colorImageView.format = colorFormat;
		colorImageView.flags = 0;
		colorImageView.subresourceRange = {};
		colorImageView.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
		colorImageView.subresourceRange.baseMipLevel = 0;
		colorImageView.subresourceRange.levelCount = 1;
		colorImageView.subresourceRange.baseArrayLayer = 0;
		colorImageView.subresourceRange.layerCount = 1;

		VK_CHECK_RESULT(vkCreateImage(device, &image, nullptr, &frameBuf->color.image));
		vkGetImageMemoryRequirements(device, frameBuf->color.image, &memReqs);
		memAlloc.allocationSize = memReqs.size;
		memAlloc.memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);
		VK_CHECK_RESULT(vkAllocateMemory(device, &memAlloc, nullptr, &frameBuf->color.mem));
		VK_CHECK_RESULT(vkBindImageMemory(device, frameBuf->color.image, frameBuf->color.mem, 0));

		colorImageView.image = frameBuf->color.image;
		VK_CHECK_RESULT(vkCreateImageView(device, &colorImageView, nullptr, &frameBuf->color.view));

		// Depth stencil attachment
		image.format = depthFormat;
		image.usage = vk::ImageUsageFlagBits::eDepthStencilAttachment;

		vk::ImageViewCreateInfo depthStencilView = vks::initializers::imageViewCreateInfo();
		depthStencilView.viewType = vk::ImageViewType::e2D;
		depthStencilView.format = depthFormat;
		depthStencilView.flags = 0;
		depthStencilView.subresourceRange = {};
		depthStencilView.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil;
		depthStencilView.subresourceRange.baseMipLevel = 0;
		depthStencilView.subresourceRange.levelCount = 1;
		depthStencilView.subresourceRange.baseArrayLayer = 0;
		depthStencilView.subresourceRange.layerCount = 1;

		VK_CHECK_RESULT(vkCreateImage(device, &image, nullptr, &frameBuf->depth.image));
		vkGetImageMemoryRequirements(device, frameBuf->depth.image, &memReqs);
		memAlloc.allocationSize = memReqs.size;
		memAlloc.memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);
		VK_CHECK_RESULT(vkAllocateMemory(device, &memAlloc, nullptr, &frameBuf->depth.mem));
		VK_CHECK_RESULT(vkBindImageMemory(device, frameBuf->depth.image, frameBuf->depth.mem, 0));

		depthStencilView.image = frameBuf->depth.image;
		VK_CHECK_RESULT(vkCreateImageView(device, &depthStencilView, nullptr, &frameBuf->depth.view));

		vk::ImageView attachments[2];
		attachments[0] = frameBuf->color.view;
		attachments[1] = frameBuf->depth.view;

		vk::FramebufferCreateInfo fbufCreateInfo = vks::initializers::framebufferCreateInfo();
		fbufCreateInfo.renderPass = offscreenPass.renderPass;
		fbufCreateInfo.attachmentCount = 2;
		fbufCreateInfo.pAttachments = attachments;
		fbufCreateInfo.width = FB_DIM;
		fbufCreateInfo.height = FB_DIM;
		fbufCreateInfo.layers = 1;

		VK_CHECK_RESULT(vkCreateFramebuffer(device, &fbufCreateInfo, nullptr, &frameBuf->framebuffer));

		// Fill a descriptor for later use in a descriptor set 
		frameBuf->descriptor.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
		frameBuf->descriptor.imageView = frameBuf->color.view;
		frameBuf->descriptor.sampler = offscreenPass.sampler;
	}

	// Prepare the offscreen framebuffers used for the vertical- and horizontal blur 
	void prepareOffscreen()
	{
		offscreenPass.width = FB_DIM;
		offscreenPass.height = FB_DIM;

		// Find a suitable depth format
		vk::Format fbDepthFormat;
		vk::Bool32 validDepthFormat = vks::tools::getSupportedDepthFormat(physicalDevice, &fbDepthFormat);
		assert(validDepthFormat);

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
		renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		renderPassInfo.attachmentCount = static_cast<uint32_t>(attchmentDescriptions.size());
		renderPassInfo.pAttachments = attchmentDescriptions.data();
		renderPassInfo.subpassCount = 1;
		renderPassInfo.pSubpasses = &subpassDescription;
		renderPassInfo.dependencyCount = static_cast<uint32_t>(dependencies.size());
		renderPassInfo.pDependencies = dependencies.data();

		VK_CHECK_RESULT(vkCreateRenderPass(device, &renderPassInfo, nullptr, &offscreenPass.renderPass));

		// Create sampler to sample from the color attachments
		vk::SamplerCreateInfo sampler = vks::initializers::samplerCreateInfo();
		sampler.magFilter = vk::Filter::eLinear;
		sampler.minFilter = vk::Filter::eLinear;
		sampler.mipmapMode = vk::SamplerMipmapMode::eLinear;
		sampler.addressModeU = vk::SamplerAddressMode::eClampToEdge;
		sampler.addressModeV = sampler.addressModeU;
		sampler.addressModeW = sampler.addressModeU;
		sampler.mipLodBias = 0.0f;
		sampler.maxAnisotropy = 1.0f;
		sampler.minLod = 0.0f;
		sampler.maxLod = 1.0f;
		sampler.borderColor = vk::BorderColor::eFloatOpaqueWhite;
		VK_CHECK_RESULT(vkCreateSampler(device, &sampler, nullptr, &offscreenPass.sampler));

		// Create two frame buffers
		prepareOffscreenFramebuffer(&offscreenPass.framebuffers[0], FB_COLOR_FORMAT, fbDepthFormat);
		prepareOffscreenFramebuffer(&offscreenPass.framebuffers[1], FB_COLOR_FORMAT, fbDepthFormat);
	}

	// Sets up the command buffer that renders the scene to the offscreen frame buffer
	// The blur method used in this example is multi pass and renders the vertical
	// blur first and then the horizontal one.
	// While it's possible to blur in one pass, this method is widely used as it
	// requires far less samples to generate the blur
	void buildOffscreenCommandBuffer()
	{
		if (offscreenPass.commandBuffer == VK_NULL_HANDLE)
		{
			offscreenPass.commandBuffer = VulkanExampleBase::createCommandBuffer(vk::CommandBufferLevel::ePrimary, false);
		}

		if (offscreenPass.semaphore == VK_NULL_HANDLE)
		{
			vk::SemaphoreCreateInfo semaphoreCreateInfo = vks::initializers::semaphoreCreateInfo();
			VK_CHECK_RESULT(vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &offscreenPass.semaphore));
		}

		vk::CommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();

		// First pass: Render glow parts of the model (separate mesh)
		// -------------------------------------------------------------------------------------------------------

		vk::ClearValue clearValues[2];
		clearValues[0].color = { { 0.0f, 0.0f, 0.0f, 1.0f } };
		clearValues[1].depthStencil = { 1.0f, 0 };

		vk::RenderPassBeginInfo renderPassBeginInfo = vks::initializers::renderPassBeginInfo();
		renderPassBeginInfo.renderPass = offscreenPass.renderPass;
		renderPassBeginInfo.framebuffer = offscreenPass.framebuffers[0].framebuffer;
		renderPassBeginInfo.renderArea.extent.width = offscreenPass.width;
		renderPassBeginInfo.renderArea.extent.height = offscreenPass.height;
		renderPassBeginInfo.clearValueCount = 2;
		renderPassBeginInfo.pClearValues = clearValues;

		VK_CHECK_RESULT(vkBeginCommandBuffer(offscreenPass.commandBuffer, &cmdBufInfo));

		vk::Viewport viewport = vks::initializers::viewport((float)offscreenPass.width, (float)offscreenPass.height, 0.0f, 1.0f);
		vkCmdSetViewport(offscreenPass.commandBuffer, 0, 1, &viewport);

		vk::Rect2D scissor = vks::initializers::rect2D(offscreenPass.width, offscreenPass.height,	0, 0);
		vkCmdSetScissor(offscreenPass.commandBuffer, 0, 1, &scissor);

		vkCmdBeginRenderPass(offscreenPass.commandBuffer, &renderPassBeginInfo, vk::SubpassContents::eInline);

		vkCmdBindDescriptorSets(offscreenPass.commandBuffer, vk::PipelineBindPoint::eGraphics, pipelineLayouts.scene, 0, 1, &descriptorSets.scene, 0, NULL);
		vkCmdBindPipeline(offscreenPass.commandBuffer, vk::PipelineBindPoint::eGraphics, pipelines.glowPass);

		vk::DeviceSize offsets[1] = { 0 };
		vkCmdBindVertexBuffers(offscreenPass.commandBuffer, VERTEX_BUFFER_BIND_ID, 1, &models.ufoGlow.vertices.buffer, offsets);
		vkCmdBindIndexBuffer(offscreenPass.commandBuffer, models.ufoGlow.indices.buffer, 0, vk::IndexType::eUint32);
		vkCmdDrawIndexed(offscreenPass.commandBuffer, models.ufoGlow.indexCount, 1, 0, 0, 0);

		vkCmdEndRenderPass(offscreenPass.commandBuffer);

		// Second pass: Render contents of the first pass into second framebuffer and apply a vertical blur
		// This is the first blur pass, the horizontal blur is applied when rendering on top of the scene
		// -------------------------------------------------------------------------------------------------------

		renderPassBeginInfo.framebuffer = offscreenPass.framebuffers[1].framebuffer;

		vkCmdBeginRenderPass(offscreenPass.commandBuffer, &renderPassBeginInfo, vk::SubpassContents::eInline);

		vkCmdBindDescriptorSets(offscreenPass.commandBuffer, vk::PipelineBindPoint::eGraphics, pipelineLayouts.blur, 0, 1, &descriptorSets.blurVert, 0, NULL);
		vkCmdBindPipeline(offscreenPass.commandBuffer, vk::PipelineBindPoint::eGraphics, pipelines.blurVert);
		vkCmdDraw(offscreenPass.commandBuffer, 3, 1, 0, 0);

		vkCmdEndRenderPass(offscreenPass.commandBuffer);

		VK_CHECK_RESULT(vkEndCommandBuffer(offscreenPass.commandBuffer));
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

			VK_CHECK_RESULT(vkBeginCommandBuffer(drawCmdBuffers[i], &cmdBufInfo));

			vkCmdBeginRenderPass(drawCmdBuffers[i], &renderPassBeginInfo, vk::SubpassContents::eInline);

			vk::Viewport viewport = vks::initializers::viewport((float)width, (float)height, 0.0f, 1.0f);
			vkCmdSetViewport(drawCmdBuffers[i], 0, 1, &viewport);

			vk::Rect2D scissor = vks::initializers::rect2D(width, height,	0, 0);
			vkCmdSetScissor(drawCmdBuffers[i], 0, 1, &scissor);

			vk::DeviceSize offsets[1] = { 0 };

			// Skybox 
			vkCmdBindDescriptorSets(drawCmdBuffers[i], vk::PipelineBindPoint::eGraphics, pipelineLayouts.scene, 0, 1, &descriptorSets.skyBox, 0, NULL);
			vkCmdBindPipeline(drawCmdBuffers[i], vk::PipelineBindPoint::eGraphics, pipelines.skyBox);

			vkCmdBindVertexBuffers(drawCmdBuffers[i], VERTEX_BUFFER_BIND_ID, 1, &models.skyBox.vertices.buffer, offsets);
			vkCmdBindIndexBuffer(drawCmdBuffers[i], models.skyBox.indices.buffer, 0, vk::IndexType::eUint32);
			vkCmdDrawIndexed(drawCmdBuffers[i], models.skyBox.indexCount, 1, 0, 0, 0);

			// 3D scene
			vkCmdBindDescriptorSets(drawCmdBuffers[i], vk::PipelineBindPoint::eGraphics, pipelineLayouts.scene, 0, 1, &descriptorSets.scene, 0, NULL);
			vkCmdBindPipeline(drawCmdBuffers[i], vk::PipelineBindPoint::eGraphics, pipelines.phongPass);

			vkCmdBindVertexBuffers(drawCmdBuffers[i], VERTEX_BUFFER_BIND_ID, 1, &models.ufo.vertices.buffer, offsets);
			vkCmdBindIndexBuffer(drawCmdBuffers[i], models.ufo.indices.buffer, 0, vk::IndexType::eUint32);
			vkCmdDrawIndexed(drawCmdBuffers[i], models.ufo.indexCount, 1, 0, 0, 0);

			// Render vertical blurred scene applying a horizontal blur
			// Render the (vertically blurred) contents of the second framebuffer and apply a horizontal blur
			// -------------------------------------------------------------------------------------------------------
			if (bloom)
			{
				vkCmdBindDescriptorSets(drawCmdBuffers[i], vk::PipelineBindPoint::eGraphics, pipelineLayouts.blur, 0, 1, &descriptorSets.blurHorz, 0, NULL);
				vkCmdBindPipeline(drawCmdBuffers[i], vk::PipelineBindPoint::eGraphics, pipelines.blurHorz);
				vkCmdDraw(drawCmdBuffers[i], 3, 1, 0, 0);
			}

			vkCmdEndRenderPass(drawCmdBuffers[i]);

			VK_CHECK_RESULT(vkEndCommandBuffer(drawCmdBuffers[i]));
		}

		if (bloom) 
		{
			buildOffscreenCommandBuffer();
		}
	}

	void loadAssets()
	{
		models.ufo.loadFromFile(getAssetPath() + "models/retroufo.dae", vertexLayout, 0.05f, vulkanDevice, queue);
		models.ufoGlow.loadFromFile(getAssetPath() + "models/retroufo_glow.dae", vertexLayout, 0.05f, vulkanDevice, queue);
		models.skyBox.loadFromFile(getAssetPath() + "models/cube.obj", vertexLayout, 1.0f, vulkanDevice, queue);
		textures.cubemap.loadFromFile(getAssetPath() + "textures/cubemap_space.ktx", vk::Format::eR8G8B8A8Unorm, vulkanDevice, queue);
	}

	void setupVertexDescriptions()
	{
		// Binding description
		// Same for all meshes used in this example
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
			vks::initializers::descriptorPoolSize(vk::DescriptorType::eUniformBuffer, 8),
			vks::initializers::descriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, 6)
		};

		vk::DescriptorPoolCreateInfo descriptorPoolInfo =
			vks::initializers::descriptorPoolCreateInfo(
				poolSizes.size(),
				poolSizes.data(),
				5);

		VK_CHECK_RESULT(vkCreateDescriptorPool(device, &descriptorPoolInfo, nullptr, &descriptorPool));
	}

	void setupDescriptorSetLayout()
	{
		std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings;
		vk::DescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo;
		vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo;

		// Fullscreen blur
		setLayoutBindings = {
			vks::initializers::descriptorSetLayoutBinding(vk::DescriptorType::eUniformBuffer, vk::ShaderStageFlagBits::eFragment, 0),			// Binding 0: Fragment shader uniform buffer
			vks::initializers::descriptorSetLayoutBinding(vk::DescriptorType::eCombinedImageSampler, vk::ShaderStageFlagBits::eFragment, 1)	// Binding 1: Fragment shader image sampler
		};
		descriptorSetLayoutCreateInfo = vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings.data(), static_cast<uint32_t>(setLayoutBindings.size()));
		VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, nullptr, &descriptorSetLayouts.blur));
		pipelineLayoutCreateInfo = vks::initializers::pipelineLayoutCreateInfo(&descriptorSetLayouts.blur, 1);
		VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr, &pipelineLayouts.blur));

		// Scene rendering
		setLayoutBindings = {			
			vks::initializers::descriptorSetLayoutBinding(vk::DescriptorType::eUniformBuffer, vk::ShaderStageFlagBits::eVertex, 0),			// Binding 0 : Vertex shader uniform buffer			
			vks::initializers::descriptorSetLayoutBinding(vk::DescriptorType::eCombinedImageSampler, vk::ShaderStageFlagBits::eFragment, 1),	// Binding 1 : Fragment shader image sampler			
			vks::initializers::descriptorSetLayoutBinding(vk::DescriptorType::eUniformBuffer, vk::ShaderStageFlagBits::eFragment, 2),			// Binding 2 : Framgnet shader image sampler
		};

		descriptorSetLayoutCreateInfo = vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings.data(), setLayoutBindings.size());
		VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, nullptr, &descriptorSetLayouts.scene));
		pipelineLayoutCreateInfo = vks::initializers::pipelineLayoutCreateInfo(&descriptorSetLayouts.scene, 1);
		VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr, &pipelineLayouts.scene));
	}

	void setupDescriptorSet()
	{
		vk::DescriptorSetAllocateInfo descriptorSetAllocInfo;
		std::vector<vk::WriteDescriptorSet> writeDescriptorSets;

		// Full screen blur
		// Vertical
		descriptorSetAllocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorPool, &descriptorSetLayouts.blur, 1);
		VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &descriptorSetAllocInfo, &descriptorSets.blurVert));
		writeDescriptorSets = {			
			vks::initializers::writeDescriptorSet(descriptorSets.blurVert, vk::DescriptorType::eUniformBuffer, 0, &uniformBuffers.blurParams.descriptor),				// Binding 0: Fragment shader uniform buffer			
			vks::initializers::writeDescriptorSet(descriptorSets.blurVert, vk::DescriptorType::eCombinedImageSampler, 1, &offscreenPass.framebuffers[0].descriptor),	// Binding 1: Fragment shader texture sampler
		};
		vkUpdateDescriptorSets(device, writeDescriptorSets.size(), writeDescriptorSets.data(), 0, NULL);
		// Horizontal
		descriptorSetAllocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorPool, &descriptorSetLayouts.blur, 1);
		VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &descriptorSetAllocInfo, &descriptorSets.blurHorz));
		writeDescriptorSets = {
			vks::initializers::writeDescriptorSet(descriptorSets.blurHorz, vk::DescriptorType::eUniformBuffer, 0, &uniformBuffers.blurParams.descriptor),				// Binding 0: Fragment shader uniform buffer			
			vks::initializers::writeDescriptorSet(descriptorSets.blurHorz, vk::DescriptorType::eCombinedImageSampler, 1, &offscreenPass.framebuffers[1].descriptor),	// Binding 1: Fragment shader texture sampler
		};
		vkUpdateDescriptorSets(device, writeDescriptorSets.size(), writeDescriptorSets.data(), 0, NULL);

		// Scene rendering
		descriptorSetAllocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorPool, &descriptorSetLayouts.scene, 1);
		VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &descriptorSetAllocInfo, &descriptorSets.scene));
		writeDescriptorSets = {			
			vks::initializers::writeDescriptorSet(descriptorSets.scene, vk::DescriptorType::eUniformBuffer, 0, &uniformBuffers.scene.descriptor)							// Binding 0: Vertex shader uniform buffer
		};
		vkUpdateDescriptorSets(device, writeDescriptorSets.size(), writeDescriptorSets.data(), 0, NULL);

		// Skybox
		VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &descriptorSetAllocInfo, &descriptorSets.skyBox));
		writeDescriptorSets = {			
			vks::initializers::writeDescriptorSet(descriptorSets.skyBox, vk::DescriptorType::eUniformBuffer, 0, &uniformBuffers.skyBox.descriptor),						// Binding 0: Vertex shader uniform buffer			
			vks::initializers::writeDescriptorSet(descriptorSets.skyBox, vk::DescriptorType::eCombinedImageSampler,	1, &textures.cubemap.descriptor),					// Binding 1: Fragment shader texture sampler
		};
		vkUpdateDescriptorSets(device, writeDescriptorSets.size(), writeDescriptorSets.data(), 0, NULL);
	}

	void preparePipelines()
	{
		vk::PipelineInputAssemblyStateCreateInfo inputAssemblyState =
			vks::initializers::pipelineInputAssemblyStateCreateInfo(
				vk::PrimitiveTopology::eTriangleList,
				0,
				VK_FALSE);

		vk::PipelineRasterizationStateCreateInfo rasterizationState =
			vks::initializers::pipelineRasterizationStateCreateInfo(
				vk::PolygonMode::eFill,
				vk::CullModeFlagBits::eNone,
				vk::FrontFace::eClockwise,
				0);

		vk::PipelineColorBlendAttachmentState blendAttachmentState =
			vks::initializers::pipelineColorBlendAttachmentState(
				0xf,
				VK_FALSE);

		vk::PipelineColorBlendStateCreateInfo colorBlendState =
			vks::initializers::pipelineColorBlendStateCreateInfo(
				1,
				&blendAttachmentState);

		vk::PipelineDepthStencilStateCreateInfo depthStencilState =
			vks::initializers::pipelineDepthStencilStateCreateInfo(
				VK_TRUE,
				VK_TRUE,
				vk::CompareOp::eLess_OR_EQUAL);

		vk::PipelineViewportStateCreateInfo viewportState =
			vks::initializers::pipelineViewportStateCreateInfo(1, 1, 0);

		vk::PipelineMultisampleStateCreateInfo multisampleState =
			vks::initializers::pipelineMultisampleStateCreateInfo(
				vk::SampleCountFlagBits::e1,
				0);

		std::vector<vk::DynamicState> dynamicStateEnables = {
			vk::DynamicState::eViewport,
			vk::DynamicState::eScissor
		};
		vk::PipelineDynamicStateCreateInfo dynamicState =
			vks::initializers::pipelineDynamicStateCreateInfo(
				dynamicStateEnables.data(),
				dynamicStateEnables.size(),
				0);

		std::array<vk::PipelineShaderStageCreateInfo, 2> shaderStages;

		vk::GraphicsPipelineCreateInfo pipelineCreateInfo =
			vks::initializers::pipelineCreateInfo(
				pipelineLayouts.blur,
				renderPass,
				0);

		pipelineCreateInfo.pInputAssemblyState = &inputAssemblyState;
		pipelineCreateInfo.pRasterizationState = &rasterizationState;
		pipelineCreateInfo.pColorBlendState = &colorBlendState;
		pipelineCreateInfo.pMultisampleState = &multisampleState;
		pipelineCreateInfo.pViewportState = &viewportState;
		pipelineCreateInfo.pDepthStencilState = &depthStencilState;
		pipelineCreateInfo.pDynamicState = &dynamicState;
		pipelineCreateInfo.stageCount = shaderStages.size();
		pipelineCreateInfo.pStages = shaderStages.data();

		// Blur pipelines
		shaderStages[0] = loadShader(getAssetPath() + "shaders/bloom/gaussblur.vert.spv", vk::ShaderStageFlagBits::eVertex);
		shaderStages[1] = loadShader(getAssetPath() + "shaders/bloom/gaussblur.frag.spv", vk::ShaderStageFlagBits::eFragment);
		// Empty vertex input state
		vk::PipelineVertexInputStateCreateInfo emptyInputState = vks::initializers::pipelineVertexInputStateCreateInfo();
		pipelineCreateInfo.pVertexInputState = &emptyInputState;
		pipelineCreateInfo.layout = pipelineLayouts.blur;
		// Additive blending
		blendAttachmentState.colorWriteMask = 0xF;
		blendAttachmentState.blendEnable = VK_TRUE;
		blendAttachmentState.colorBlendOp = vk::BlendOp::eAdd;
		blendAttachmentState.srcColorBlendFactor = vk::BlendFactor::eOne;
		blendAttachmentState.dstColorBlendFactor = vk::BlendFactor::eOne;
		blendAttachmentState.alphaBlendOp = vk::BlendOp::eAdd;
		blendAttachmentState.srcAlphaBlendFactor = vk::BlendFactor::eSrcAlpha;
		blendAttachmentState.dstAlphaBlendFactor = vk::BlendFactor::eDstAlpha;

		// Use specialization constants to select between horizontal and vertical blur
		uint32_t blurdirection = 0;
		vk::SpecializationMapEntry specializationMapEntry = vks::initializers::specializationMapEntry(0, 0, sizeof(uint32_t));
		vk::SpecializationInfo specializationInfo = vks::initializers::specializationInfo(1, &specializationMapEntry, sizeof(uint32_t), &blurdirection);
		shaderStages[1].pSpecializationInfo = &specializationInfo;
		// Vertical blur pipeline
		pipelineCreateInfo.renderPass = offscreenPass.renderPass;
		VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCreateInfo, nullptr, &pipelines.blurVert));
		// Horizontal blur pipeline
		blurdirection = 1;
		pipelineCreateInfo.renderPass = renderPass;
		VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCreateInfo, nullptr, &pipelines.blurHorz));

		// Phong pass (3D model)
		pipelineCreateInfo.layout = pipelineLayouts.scene;
		pipelineCreateInfo.pVertexInputState = &vertices.inputState;
		shaderStages[0] = loadShader(getAssetPath() + "shaders/bloom/phongpass.vert.spv", vk::ShaderStageFlagBits::eVertex);
		shaderStages[1] = loadShader(getAssetPath() + "shaders/bloom/phongpass.frag.spv", vk::ShaderStageFlagBits::eFragment);
		blendAttachmentState.blendEnable = VK_FALSE;
		depthStencilState.depthWriteEnable = VK_TRUE;
		rasterizationState.cullMode = vk::CullModeFlagBits::eBack;
		pipelineCreateInfo.renderPass = renderPass;
		VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCreateInfo, nullptr, &pipelines.phongPass));

		// Color only pass (offscreen blur base)
		shaderStages[0] = loadShader(getAssetPath() + "shaders/bloom/colorpass.vert.spv", vk::ShaderStageFlagBits::eVertex);
		shaderStages[1] = loadShader(getAssetPath() + "shaders/bloom/colorpass.frag.spv", vk::ShaderStageFlagBits::eFragment);
		pipelineCreateInfo.renderPass = offscreenPass.renderPass;
		VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCreateInfo, nullptr, &pipelines.glowPass));

		// Skybox (cubemap)
		shaderStages[0] = loadShader(getAssetPath() + "shaders/bloom/skybox.vert.spv", vk::ShaderStageFlagBits::eVertex);
		shaderStages[1] = loadShader(getAssetPath() + "shaders/bloom/skybox.frag.spv", vk::ShaderStageFlagBits::eFragment);
		depthStencilState.depthWriteEnable = VK_FALSE;
		rasterizationState.cullMode = vk::CullModeFlagBits::eFront;
		pipelineCreateInfo.renderPass = renderPass;
		VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCreateInfo, nullptr, &pipelines.skyBox));
	}

	// Prepare and initialize uniform buffer containing shader uniforms
	void prepareUniformBuffers()
	{
		// Phong and color pass vertex shader uniform buffer
		VK_CHECK_RESULT(vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eUniformBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			&uniformBuffers.scene,
			sizeof(ubos.scene)));

		// Blur parameters uniform buffers
		VK_CHECK_RESULT(vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eUniformBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			&uniformBuffers.blurParams,
			sizeof(ubos.blurParams)));

		// Skybox
		VK_CHECK_RESULT(vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eUniformBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			&uniformBuffers.skyBox,
			sizeof(ubos.skyBox)));

		// Map persistent
		VK_CHECK_RESULT(uniformBuffers.scene.map());
		VK_CHECK_RESULT(uniformBuffers.blurParams.map());
		VK_CHECK_RESULT(uniformBuffers.skyBox.map());

		// Intialize uniform buffers
		updateUniformBuffersScene();
		updateUniformBuffersBlur();
	}

	// Update uniform buffers for rendering the 3D scene
	void updateUniformBuffersScene()
	{
		// UFO
		ubos.scene.projection = camera.matrices.perspective;
		ubos.scene.view = camera.matrices.view;

		ubos.scene.model = glm::translate(glm::mat4(), glm::vec3(sin(glm::radians(timer * 360.0f)) * 0.25f, -1.0f, cos(glm::radians(timer * 360.0f)) * 0.25f) + cameraPos);
		ubos.scene.model = glm::rotate(ubos.scene.model, -sinf(glm::radians(timer * 360.0f)) * 0.15f, glm::vec3(1.0f, 0.0f, 0.0f));
		ubos.scene.model = glm::rotate(ubos.scene.model, glm::radians(timer * 360.0f), glm::vec3(0.0f, 1.0f, 0.0f));

		memcpy(uniformBuffers.scene.mapped, &ubos.scene, sizeof(ubos.scene));

		// Skybox
		ubos.skyBox.projection = glm::perspective(glm::radians(45.0f), (float)width / (float)height, 0.1f, 256.0f);
		ubos.skyBox.view = glm::mat4(glm::mat3(camera.matrices.view));
		ubos.skyBox.model = glm::mat4();

		memcpy(uniformBuffers.skyBox.mapped, &ubos.skyBox, sizeof(ubos.skyBox));
	}

	// Update blur pass parameter uniform buffer
	void updateUniformBuffersBlur()
	{
		memcpy(uniformBuffers.blurParams.mapped, &ubos.blurParams, sizeof(ubos.blurParams));
	}

	void draw()
	{
		VulkanExampleBase::prepareFrame();

		// The scene render command buffer has to wait for the offscreen rendering to be finished before we can use the framebuffer 
		// color image for sampling during final rendering
		// To ensure this we use a dedicated offscreen synchronization semaphore that will be signaled when offscreen rendering has been finished
		// This is necessary as an implementation may start both command buffers at the same time, there is no guarantee
		// that command buffers will be executed in the order they have been submitted by the application

		// Offscreen rendering

		// Wait for swap chain presentation to finish
		submitInfo.pWaitSemaphores = &semaphores.presentComplete;
		// Signal ready with offscreen semaphore
		submitInfo.pSignalSemaphores = &offscreenPass.semaphore;

		// Submit work
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &offscreenPass.commandBuffer;
		VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));

		// Scene rendering

		// Wait for offscreen semaphore
		submitInfo.pWaitSemaphores = &offscreenPass.semaphore;
		// Signal ready with render complete semaphpre
		submitInfo.pSignalSemaphores = &semaphores.renderComplete;

		// Submit work
		submitInfo.pCommandBuffers = &drawCmdBuffers[currentBuffer];
		VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));

		VulkanExampleBase::submitFrame();
	}

	void prepare()
	{
		VulkanExampleBase::prepare();
		loadAssets();
		setupVertexDescriptions();
		prepareUniformBuffers();
		prepareOffscreen();
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
			updateUniformBuffersScene();
		}
	}

	virtual void viewChanged()
	{
		updateUniformBuffersScene();
	}

	void changeBlurScale(float delta)
	{
		ubos.blurParams.blurScale += delta;
		updateUniformBuffersBlur();
	}

	void toggleBloom()
	{
		bloom = !bloom;
		reBuildCommandBuffers();
	}

	virtual void keyPressed(uint32_t keyCode)
	{
		switch (keyCode)
		{
		case KEY_KPADD:
		case GAMEPAD_BUTTON_R1:
			changeBlurScale(0.25f);
			break;
		case KEY_KPSUB:
		case GAMEPAD_BUTTON_L1:
			changeBlurScale(-0.25f);
			break;
		case KEY_B:
		case GAMEPAD_BUTTON_A:
			toggleBloom();
			break;
		}
	}

	virtual void getOverlayText(VulkanTextOverlay *textOverlay)
	{
#if defined(__ANDROID__)
		textOverlay->addText("Press \"L1/R1\" to change blur scale", 5.0f, 85.0f, VulkanTextOverlay::alignLeft);
		textOverlay->addText("Press \"Button A\" to toggle bloom", 5.0f, 105.0f, VulkanTextOverlay::alignLeft);
#else
		textOverlay->addText("Press \"NUMPAD +/-\" to change blur scale", 5.0f, 85.0f, VulkanTextOverlay::alignLeft);
		textOverlay->addText("Press \"B\" to toggle bloom", 5.0f, 105.0f, VulkanTextOverlay::alignLeft);
#endif
	}
};

VULKAN_EXAMPLE_MAIN()
