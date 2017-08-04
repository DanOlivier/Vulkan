/*
* Vulkan Example - HDR
*
* Note: Requires the separate asset pack (see data/README.md)
*
* Copyright (C) 2016-2017 by Sascha Willems - www.saschawillems.de
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

#include <gli/gli.hpp>

#include <vulkan/vulkan.hpp>
#include "vulkanexamplebase.h"
#include "VulkanBuffer.hpp"
#include "VulkanModel.hpp"
#include "VulkanTexture.hpp"

#define ENABLE_VALIDATION false

class VulkanExample : public VulkanExampleBase
{
public:
	bool bloom = true;
	bool displaySkybox = true;

	// Vertex layout for the models
	vks::VertexLayout vertexLayout = vks::VertexLayout({
		vks::VERTEX_COMPONENT_POSITION,
		vks::VERTEX_COMPONENT_NORMAL,
		vks::VERTEX_COMPONENT_UV,
	});

	struct {
		vks::TextureCubeMap envmap;
	} textures;

	struct Models {
		vks::Model skybox;
		std::vector<vks::Model> objects;
		uint32_t objectIndex = 1;
	} models;

	struct {
		vks::Buffer matrices;
		vks::Buffer params;
	} uniformBuffers;

	struct UBOVS {
		glm::mat4 projection;
		glm::mat4 modelview;
	} uboVS;

	struct UBOParams {
		float exposure = 1.0f;
	} uboParams;

	struct {
		vk::Pipeline skybox;
		vk::Pipeline reflect;
		vk::Pipeline composition;
		vk::Pipeline bloom[2];
	} pipelines;

	struct {
		vk::PipelineLayout models;
		vk::PipelineLayout composition;
		vk::PipelineLayout bloomFilter;
	} pipelineLayouts;

	struct {
		vk::DescriptorSet object;
		vk::DescriptorSet skybox;
		vk::DescriptorSet composition;
		vk::DescriptorSet bloomFilter;
	} descriptorSets;

	struct {
		vk::DescriptorSetLayout models;
		vk::DescriptorSetLayout composition;
		vk::DescriptorSetLayout bloomFilter;
	} descriptorSetLayouts;

	// Framebuffer for offscreen rendering
	struct FrameBufferAttachment {
		vk::Image image;
		vk::DeviceMemory mem;
		vk::ImageView view;
		vk::Format format;
		void destroy(vk::Device device)
		{
			vkDestroyImageView(device, view, nullptr);
			vkDestroyImage(device, image, nullptr);
			vkFreeMemory(device, mem, nullptr);
		}
	};
	struct FrameBuffer {
		int32_t width, height;
		vk::Framebuffer frameBuffer;
		FrameBufferAttachment color[2];
		FrameBufferAttachment depth;
		vk::RenderPass renderPass;
		vk::Sampler sampler;
		vk::CommandBuffer cmdBuffer = VK_NULL_HANDLE;
		vk::Semaphore semaphore = VK_NULL_HANDLE;
	} offscreen;

	struct {
		int32_t width, height;
		vk::Framebuffer frameBuffer;
		FrameBufferAttachment color[1];
		vk::RenderPass renderPass;
		vk::Sampler sampler;
	} filterPass;

	VulkanExample() : VulkanExampleBase(ENABLE_VALIDATION)
	{
		title = "Vulkan Example - HDR rendering";
		enableTextOverlay = true;
		camera.type = Camera::CameraType::lookat;
		camera.setPosition(glm::vec3(0.0f, 0.0f, -4.0f));
		camera.setRotation(glm::vec3(0.0f, 180.0f, 0.0f));
		camera.setPerspective(60.0f, (float)width / (float)height, 0.1f, 256.0f);
	}

	~VulkanExample()
	{
		vkDestroyPipeline(device, pipelines.skybox, nullptr);
		vkDestroyPipeline(device, pipelines.reflect, nullptr);
		vkDestroyPipeline(device, pipelines.composition, nullptr);
		vkDestroyPipeline(device, pipelines.bloom[0], nullptr);
		vkDestroyPipeline(device, pipelines.bloom[1], nullptr);

		vkDestroyPipelineLayout(device, pipelineLayouts.models, nullptr);
		vkDestroyPipelineLayout(device, pipelineLayouts.composition, nullptr);
		vkDestroyPipelineLayout(device, pipelineLayouts.bloomFilter, nullptr);

		vkDestroyDescriptorSetLayout(device, descriptorSetLayouts.models, nullptr);
		vkDestroyDescriptorSetLayout(device, descriptorSetLayouts.composition, nullptr);
		vkDestroyDescriptorSetLayout(device, descriptorSetLayouts.bloomFilter, nullptr);

		vkDestroySemaphore(device, offscreen.semaphore, nullptr);

		vkDestroyRenderPass(device, offscreen.renderPass, nullptr);
		vkDestroyRenderPass(device, filterPass.renderPass, nullptr);

		vkDestroyFramebuffer(device, offscreen.frameBuffer, nullptr);
		vkDestroyFramebuffer(device, filterPass.frameBuffer, nullptr);

		vkDestroySampler(device, offscreen.sampler, nullptr);
		vkDestroySampler(device, filterPass.sampler, nullptr);

		offscreen.depth.destroy(device);
		offscreen.color[0].destroy(device);
		offscreen.color[1].destroy(device);

		filterPass.color[0].destroy(device);

		for (auto& model : models.objects) {
			model.destroy();
		}
		models.skybox.destroy();
		uniformBuffers.matrices.destroy();
		uniformBuffers.params.destroy();
		textures.envmap.destroy();
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
		clearValues[0].color = defaultClearColor;
		clearValues[0].color = { { 0.0f, 0.0f, 0.0f, 0.0f } };;
		clearValues[1].depthStencil = { 1.0f, 0 };

		vk::RenderPassBeginInfo renderPassBeginInfo = vks::initializers::renderPassBeginInfo();
		renderPassBeginInfo.renderPass = renderPass;
		renderPassBeginInfo.renderArea.offset.x = 0;
		renderPassBeginInfo.renderArea.offset.y = 0;
		renderPassBeginInfo.clearValueCount = 2;
		renderPassBeginInfo.pClearValues = clearValues;

		vk::Viewport viewport;
		vk::Rect2D scissor;
		//vk::DeviceSize offsets[1] = { 0 };

		for (uint32_t i = 0; i < drawCmdBuffers.size(); ++i)
		{

			VK_CHECK_RESULT(vkBeginCommandBuffer(drawCmdBuffers[i], &cmdBufInfo));

			// Bloom filter
			renderPassBeginInfo.framebuffer = filterPass.frameBuffer;
			renderPassBeginInfo.renderPass = filterPass.renderPass;
			renderPassBeginInfo.clearValueCount = 1;
			renderPassBeginInfo.renderArea.extent.width = filterPass.width;
			renderPassBeginInfo.renderArea.extent.height = filterPass.height;

			viewport = vks::initializers::viewport((float)filterPass.width, (float)filterPass.height, 0.0f, 1.0f);
			scissor = vks::initializers::rect2D(filterPass.width, filterPass.height, 0, 0);

			vkCmdBeginRenderPass(drawCmdBuffers[i], &renderPassBeginInfo, vk::SubpassContents::eInline);

			vkCmdSetViewport(drawCmdBuffers[i], 0, 1, &viewport);
			vkCmdSetScissor(drawCmdBuffers[i], 0, 1, &scissor);

			vkCmdBindDescriptorSets(drawCmdBuffers[i], vk::PipelineBindPoint::eGraphics, pipelineLayouts.bloomFilter, 0, 1, &descriptorSets.bloomFilter, 0, NULL);

			vkCmdBindPipeline(drawCmdBuffers[i], vk::PipelineBindPoint::eGraphics, pipelines.bloom[1]);
			vkCmdDraw(drawCmdBuffers[i], 3, 1, 0, 0);

			vkCmdEndRenderPass(drawCmdBuffers[i]);

			viewport = vks::initializers::viewport((float)width, (float)height, 0.0f, 1.0f);
			scissor = vks::initializers::rect2D(width, height, 0, 0);

			// Final composition
			renderPassBeginInfo.framebuffer = frameBuffers[i];
			renderPassBeginInfo.renderPass = renderPass;
			renderPassBeginInfo.clearValueCount = 2;
			renderPassBeginInfo.renderArea.extent.width = width;
			renderPassBeginInfo.renderArea.extent.height = height;

			vkCmdBeginRenderPass(drawCmdBuffers[i], &renderPassBeginInfo, vk::SubpassContents::eInline);

			vkCmdSetViewport(drawCmdBuffers[i], 0, 1, &viewport);
			vkCmdSetScissor(drawCmdBuffers[i], 0, 1, &scissor);

			vkCmdBindDescriptorSets(drawCmdBuffers[i], vk::PipelineBindPoint::eGraphics, pipelineLayouts.composition, 0, 1, &descriptorSets.composition, 0, NULL);

			// Scene
			vkCmdBindPipeline(drawCmdBuffers[i], vk::PipelineBindPoint::eGraphics, pipelines.composition);
			vkCmdDraw(drawCmdBuffers[i], 3, 1, 0, 0);

			// Bloom
			if (bloom)
			{
				vkCmdBindPipeline(drawCmdBuffers[i], vk::PipelineBindPoint::eGraphics, pipelines.bloom[0]);
				vkCmdDraw(drawCmdBuffers[i], 3, 1, 0, 0);
			}

			vkCmdEndRenderPass(drawCmdBuffers[i]);

			VK_CHECK_RESULT(vkEndCommandBuffer(drawCmdBuffers[i]));
		}
	}

	void createAttachment(vk::Format format, vk::ImageUsageFlagBits usage, FrameBufferAttachment *attachment)
	{
		vk::ImageAspectFlags aspectMask = 0;
		//vk::ImageLayout imageLayout;

		attachment->format = format;

		if (usage & vk::ImageUsageFlagBits::eColorAttachment)
		{
			aspectMask = vk::ImageAspectFlagBits::eColor;
			//imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
		}
		if (usage & vk::ImageUsageFlagBits::eDepthStencilAttachment)
		{
			aspectMask = vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil;
			//imageLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal;
		}

		assert(aspectMask > 0);

		vk::ImageCreateInfo image = vks::initializers::imageCreateInfo();
		image.imageType = vk::ImageType::e2D;
		image.format = format;
		image.extent.width = offscreen.width;
		image.extent.height = offscreen.height;
		image.extent.depth = 1;
		image.mipLevels = 1;
		image.arrayLayers = 1;
		image.samples = vk::SampleCountFlagBits::e1;
		image.tiling = vk::ImageTiling::eOptimal;
		image.usage = usage | vk::ImageUsageFlagBits::eSampled;

		vk::MemoryAllocateInfo memAlloc = vks::initializers::memoryAllocateInfo();
		vk::MemoryRequirements memReqs;

		VK_CHECK_RESULT(vkCreateImage(device, &image, nullptr, &attachment->image));
		vkGetImageMemoryRequirements(device, attachment->image, &memReqs);
		memAlloc.allocationSize = memReqs.size;
		memAlloc.memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);
		VK_CHECK_RESULT(vkAllocateMemory(device, &memAlloc, nullptr, &attachment->mem));
		VK_CHECK_RESULT(vkBindImageMemory(device, attachment->image, attachment->mem, 0));

		vk::ImageViewCreateInfo imageView = vks::initializers::imageViewCreateInfo();
		imageView.viewType = vk::ImageViewType::e2D;
		imageView.format = format;
		imageView.subresourceRange = {};
		imageView.subresourceRange.aspectMask = aspectMask;
		imageView.subresourceRange.baseMipLevel = 0;
		imageView.subresourceRange.levelCount = 1;
		imageView.subresourceRange.baseArrayLayer = 0;
		imageView.subresourceRange.layerCount = 1;
		imageView.image = attachment->image;
		VK_CHECK_RESULT(vkCreateImageView(device, &imageView, nullptr, &attachment->view));
	}

	// Prepare a new framebuffer and attachments for offscreen rendering (G-Buffer)
	void prepareoffscreenfer()
	{
		{
			offscreen.width = width;
			offscreen.height = height;

			// Color attachments

			// Two floating point color buffers
			createAttachment(vk::Format::eR32G32B32A32Sfloat, vk::ImageUsageFlagBits::eColorAttachment, &offscreen.color[0]);
			createAttachment(vk::Format::eR32G32B32A32Sfloat, vk::ImageUsageFlagBits::eColorAttachment, &offscreen.color[1]);
			// Depth attachment
			createAttachment(depthFormat, vk::ImageUsageFlagBits::eDepthStencilAttachment, &offscreen.depth);

			// Set up separate renderpass with references to the colorand depth attachments
			std::array<vk::AttachmentDescription, 3> attachmentDescs = {};

			// Init attachment properties
			for (uint32_t i = 0; i < 3; ++i)
			{
				attachmentDescs[i].samples = vk::SampleCountFlagBits::e1;
				attachmentDescs[i].loadOp = vk::AttachmentLoadOp::eClear;
				attachmentDescs[i].storeOp = vk::AttachmentStoreOp::eStore;
				attachmentDescs[i].stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
				attachmentDescs[i].stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
				if (i == 2)
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
			attachmentDescs[0].format = offscreen.color[0].format;
			attachmentDescs[1].format = offscreen.color[1].format;
			attachmentDescs[2].format = offscreen.depth.format;

			std::vector<vk::AttachmentReference> colorReferences;
			colorReferences.push_back({ 0, vk::ImageLayout::eColorAttachmentOptimal });
			colorReferences.push_back({ 1, vk::ImageLayout::eColorAttachmentOptimal });

			vk::AttachmentReference depthReference = {};
			depthReference.attachment = 2;
			depthReference.layout = vk::ImageLayout::eDepthStencilAttachmentOptimal;

			vk::SubpassDescription subpass = {};
			subpass.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
			subpass.pColorAttachments = colorReferences.data();
			subpass.colorAttachmentCount = 2;
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
			renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
			renderPassInfo.pAttachments = attachmentDescs.data();
			renderPassInfo.attachmentCount = static_cast<uint32_t>(attachmentDescs.size());
			renderPassInfo.subpassCount = 1;
			renderPassInfo.pSubpasses = &subpass;
			renderPassInfo.dependencyCount = 2;
			renderPassInfo.pDependencies = dependencies.data();

			VK_CHECK_RESULT(vkCreateRenderPass(device, &renderPassInfo, nullptr, &offscreen.renderPass));

			std::array<vk::ImageView, 3> attachments;
			attachments[0] = offscreen.color[0].view;
			attachments[1] = offscreen.color[1].view;
			attachments[2] = offscreen.depth.view;

			vk::FramebufferCreateInfo fbufCreateInfo = {};
			fbufCreateInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
			fbufCreateInfo.pNext = NULL;
			fbufCreateInfo.renderPass = offscreen.renderPass;
			fbufCreateInfo.pAttachments = attachments.data();
			fbufCreateInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
			fbufCreateInfo.width = offscreen.width;
			fbufCreateInfo.height = offscreen.height;
			fbufCreateInfo.layers = 1;
			VK_CHECK_RESULT(vkCreateFramebuffer(device, &fbufCreateInfo, nullptr, &offscreen.frameBuffer));

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
			VK_CHECK_RESULT(vkCreateSampler(device, &sampler, nullptr, &offscreen.sampler));
		}

		// Bloom separable filter pass
		{
			filterPass.width = width;
			filterPass.height = height;

			// Color attachments

			// Two floating point color buffers
			createAttachment(vk::Format::eR32G32B32A32Sfloat, vk::ImageUsageFlagBits::eColorAttachment, &filterPass.color[0]);

			// Set up separate renderpass with references to the colorand depth attachments
			std::array<vk::AttachmentDescription, 1> attachmentDescs = {};

			// Init attachment properties
			attachmentDescs[0].samples = vk::SampleCountFlagBits::e1;
			attachmentDescs[0].loadOp = vk::AttachmentLoadOp::eClear;
			attachmentDescs[0].storeOp = vk::AttachmentStoreOp::eStore;
			attachmentDescs[0].stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
			attachmentDescs[0].stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
			attachmentDescs[0].initialLayout = vk::ImageLayout::eUndefined;
			attachmentDescs[0].finalLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
			attachmentDescs[0].format = filterPass.color[0].format;

			std::vector<vk::AttachmentReference> colorReferences;
			colorReferences.push_back({ 0, vk::ImageLayout::eColorAttachmentOptimal });

			vk::SubpassDescription subpass = {};
			subpass.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
			subpass.pColorAttachments = colorReferences.data();
			subpass.colorAttachmentCount = 1;

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
			renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
			renderPassInfo.pAttachments = attachmentDescs.data();
			renderPassInfo.attachmentCount = static_cast<uint32_t>(attachmentDescs.size());
			renderPassInfo.subpassCount = 1;
			renderPassInfo.pSubpasses = &subpass;
			renderPassInfo.dependencyCount = 2;
			renderPassInfo.pDependencies = dependencies.data();

			VK_CHECK_RESULT(vkCreateRenderPass(device, &renderPassInfo, nullptr, &filterPass.renderPass));

			std::array<vk::ImageView, 1> attachments;
			attachments[0] = filterPass.color[0].view;

			vk::FramebufferCreateInfo fbufCreateInfo = {};
			fbufCreateInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
			fbufCreateInfo.pNext = NULL;
			fbufCreateInfo.renderPass = filterPass.renderPass;
			fbufCreateInfo.pAttachments = attachments.data();
			fbufCreateInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
			fbufCreateInfo.width = filterPass.width;
			fbufCreateInfo.height = filterPass.height;
			fbufCreateInfo.layers = 1;
			VK_CHECK_RESULT(vkCreateFramebuffer(device, &fbufCreateInfo, nullptr, &filterPass.frameBuffer));

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
			VK_CHECK_RESULT(vkCreateSampler(device, &sampler, nullptr, &filterPass.sampler));
		}
	}

	// Build command buffer for rendering the scene to the offscreen frame buffer attachments
	void buildDeferredCommandBuffer()
	{
		if (offscreen.cmdBuffer == VK_NULL_HANDLE)
		{
			offscreen.cmdBuffer = VulkanExampleBase::createCommandBuffer(vk::CommandBufferLevel::ePrimary, false);
		}

		// Create a semaphore used to synchronize offscreen rendering and usage
		if (offscreen.semaphore == VK_NULL_HANDLE)
		{
			vk::SemaphoreCreateInfo semaphoreCreateInfo = vks::initializers::semaphoreCreateInfo();
			VK_CHECK_RESULT(vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &offscreen.semaphore));
		}

		vk::CommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();

		// Clear values for all attachments written in the fragment sahder
		std::array<vk::ClearValue, 3> clearValues;
		clearValues[0].color = { { 0.0f, 0.0f, 0.0f, 0.0f } };
		clearValues[1].color = { { 0.0f, 0.0f, 0.0f, 0.0f } };
		clearValues[2].depthStencil = { 1.0f, 0 };

		vk::RenderPassBeginInfo renderPassBeginInfo = vks::initializers::renderPassBeginInfo();
		renderPassBeginInfo.renderPass = offscreen.renderPass;
		renderPassBeginInfo.framebuffer = offscreen.frameBuffer;
		renderPassBeginInfo.renderArea.extent.width = offscreen.width;
		renderPassBeginInfo.renderArea.extent.height = offscreen.height;
		renderPassBeginInfo.clearValueCount = 3;
		renderPassBeginInfo.pClearValues = clearValues.data();

		VK_CHECK_RESULT(vkBeginCommandBuffer(offscreen.cmdBuffer, &cmdBufInfo));

		vkCmdBeginRenderPass(offscreen.cmdBuffer, &renderPassBeginInfo, vk::SubpassContents::eInline);

		vk::Viewport viewport = vks::initializers::viewport((float)offscreen.width, (float)offscreen.height, 0.0f, 1.0f);
		vkCmdSetViewport(offscreen.cmdBuffer, 0, 1, &viewport);

		vk::Rect2D scissor = vks::initializers::rect2D(offscreen.width, offscreen.height, 0, 0);
		vkCmdSetScissor(offscreen.cmdBuffer, 0, 1, &scissor);

		vk::DeviceSize offsets[1] = { 0 };

		// Skybox
		if (displaySkybox)
		{
			vkCmdBindDescriptorSets(offscreen.cmdBuffer, vk::PipelineBindPoint::eGraphics, pipelineLayouts.models, 0, 1, &descriptorSets.skybox, 0, NULL);
			vkCmdBindVertexBuffers(offscreen.cmdBuffer, 0, 1, &models.skybox.vertices.buffer, offsets);
			vkCmdBindIndexBuffer(offscreen.cmdBuffer, models.skybox.indices.buffer, 0, vk::IndexType::eUint32);
			vkCmdBindPipeline(offscreen.cmdBuffer, vk::PipelineBindPoint::eGraphics, pipelines.skybox);
			vkCmdDrawIndexed(offscreen.cmdBuffer, models.skybox.indexCount, 1, 0, 0, 0);
		}

		// 3D object
		vkCmdBindDescriptorSets(offscreen.cmdBuffer, vk::PipelineBindPoint::eGraphics, pipelineLayouts.models, 0, 1, &descriptorSets.object, 0, NULL);
		vkCmdBindVertexBuffers(offscreen.cmdBuffer, 0, 1, &models.objects[models.objectIndex].vertices.buffer, offsets);
		vkCmdBindIndexBuffer(offscreen.cmdBuffer, models.objects[models.objectIndex].indices.buffer, 0, vk::IndexType::eUint32);
		vkCmdBindPipeline(offscreen.cmdBuffer, vk::PipelineBindPoint::eGraphics, pipelines.reflect);
		vkCmdDrawIndexed(offscreen.cmdBuffer, models.objects[models.objectIndex].indexCount, 1, 0, 0, 0);

		vkCmdEndRenderPass(offscreen.cmdBuffer);

		VK_CHECK_RESULT(vkEndCommandBuffer(offscreen.cmdBuffer));
	}

	void loadAssets()
	{
		// Models
		models.skybox.loadFromFile(getAssetPath() + "models/cube.obj", vertexLayout, 0.05f, vulkanDevice, queue);
		std::vector<std::string> filenames = { "geosphere.obj", "teapot.dae", "torusknot.obj", "venus.fbx" };
		for (auto file : filenames) {
			vks::Model model;
			model.loadFromFile(getAssetPath() + "models/" + file, vertexLayout, 0.05f * (file == "venus.fbx" ? 3.0f : 1.0f), vulkanDevice, queue);
			models.objects.push_back(model);
		}
		// Load HDR cube map
		textures.envmap.loadFromFile(getAssetPath() + "textures/hdr/uffizi_cube.ktx", vk::Format::eR16G16B16A16Sfloat, vulkanDevice, queue);			
	}

	void setupDescriptorPool()
	{
		std::vector<vk::DescriptorPoolSize> poolSizes = {
			vks::initializers::descriptorPoolSize(vk::DescriptorType::eUniformBuffer, 4),
			vks::initializers::descriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, 6)
		};
		uint32_t numDescriptorSets = 4;
		vk::DescriptorPoolCreateInfo descriptorPoolInfo = 
			vks::initializers::descriptorPoolCreateInfo(static_cast<uint32_t>(poolSizes.size()), poolSizes.data(), numDescriptorSets);
		VK_CHECK_RESULT(vkCreateDescriptorPool(device, &descriptorPoolInfo, nullptr, &descriptorPool));
	}

	void setupDescriptorSetLayout()
	{
		std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings = {
			vks::initializers::descriptorSetLayoutBinding(vk::DescriptorType::eUniformBuffer, vk::ShaderStageFlagBits::eVertex, 0),
			vks::initializers::descriptorSetLayoutBinding(vk::DescriptorType::eCombinedImageSampler, vk::ShaderStageFlagBits::eFragment, 1),
			vks::initializers::descriptorSetLayoutBinding(vk::DescriptorType::eUniformBuffer, vk::ShaderStageFlagBits::eFragment, 2),
		};

		vk::DescriptorSetLayoutCreateInfo descriptorLayoutInfo = 
			vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings.data(), static_cast<uint32_t>(setLayoutBindings.size()));

		VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorLayoutInfo, nullptr, &descriptorSetLayouts.models));

		vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo =
			vks::initializers::pipelineLayoutCreateInfo(
				&descriptorSetLayouts.models,
				1);

		VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr, &pipelineLayouts.models));

		// Bloom filter
		setLayoutBindings = {
			vks::initializers::descriptorSetLayoutBinding(vk::DescriptorType::eCombinedImageSampler, vk::ShaderStageFlagBits::eFragment, 0),
			vks::initializers::descriptorSetLayoutBinding(vk::DescriptorType::eCombinedImageSampler, vk::ShaderStageFlagBits::eFragment, 1),
		};

		descriptorLayoutInfo = vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings.data(), static_cast<uint32_t>(setLayoutBindings.size()));
		VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorLayoutInfo, nullptr, &descriptorSetLayouts.bloomFilter));

		pipelineLayoutCreateInfo = vks::initializers::pipelineLayoutCreateInfo(&descriptorSetLayouts.bloomFilter, 1);
		VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr, &pipelineLayouts.bloomFilter));

		// G-Buffer composition
		setLayoutBindings = {
			vks::initializers::descriptorSetLayoutBinding(vk::DescriptorType::eCombinedImageSampler, vk::ShaderStageFlagBits::eFragment, 0),
			vks::initializers::descriptorSetLayoutBinding(vk::DescriptorType::eCombinedImageSampler, vk::ShaderStageFlagBits::eFragment, 1),
		};

		descriptorLayoutInfo = vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings.data(), static_cast<uint32_t>(setLayoutBindings.size()));
		VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorLayoutInfo, nullptr, &descriptorSetLayouts.composition));

		pipelineLayoutCreateInfo = vks::initializers::pipelineLayoutCreateInfo(&descriptorSetLayouts.composition, 1);
		VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr, &pipelineLayouts.composition));
	}

	void setupDescriptorSets()
	{
		vk::DescriptorSetAllocateInfo allocInfo =
			vks::initializers::descriptorSetAllocateInfo(
				descriptorPool,
				&descriptorSetLayouts.models,
				1);

		// 3D object descriptor set
		VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &descriptorSets.object));

		std::vector<vk::WriteDescriptorSet> writeDescriptorSets = {
			vks::initializers::writeDescriptorSet(descriptorSets.object, vk::DescriptorType::eUniformBuffer, 0, &uniformBuffers.matrices.descriptor),
			vks::initializers::writeDescriptorSet(descriptorSets.object, vk::DescriptorType::eCombinedImageSampler, 1, &textures.envmap.descriptor),
			vks::initializers::writeDescriptorSet(descriptorSets.object, vk::DescriptorType::eUniformBuffer, 2, &uniformBuffers.params.descriptor),
		};
		vkUpdateDescriptorSets(device, static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, NULL);

		// Sky box descriptor set
		VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &descriptorSets.skybox));

		writeDescriptorSets = {
			vks::initializers::writeDescriptorSet(descriptorSets.skybox, vk::DescriptorType::eUniformBuffer, 0,&uniformBuffers.matrices.descriptor),
			vks::initializers::writeDescriptorSet(descriptorSets.skybox, vk::DescriptorType::eCombinedImageSampler, 1, &textures.envmap.descriptor),
			vks::initializers::writeDescriptorSet(descriptorSets.skybox, vk::DescriptorType::eUniformBuffer, 2, &uniformBuffers.params.descriptor),
		};
		vkUpdateDescriptorSets(device, static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, NULL);

		// Bloom filter 
		allocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorPool, &descriptorSetLayouts.bloomFilter, 1);
		VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &descriptorSets.bloomFilter));

		std::vector<vk::DescriptorImageInfo> colorDescriptors = {
			vks::initializers::descriptorImageInfo(offscreen.sampler, offscreen.color[0].view, vk::ImageLayout::eShaderReadOnlyOptimal),
			vks::initializers::descriptorImageInfo(offscreen.sampler, offscreen.color[1].view, vk::ImageLayout::eShaderReadOnlyOptimal),
		};

		writeDescriptorSets = {
			vks::initializers::writeDescriptorSet(descriptorSets.bloomFilter, vk::DescriptorType::eCombinedImageSampler, 0, &colorDescriptors[0]),
			vks::initializers::writeDescriptorSet(descriptorSets.bloomFilter, vk::DescriptorType::eCombinedImageSampler, 1, &colorDescriptors[1]),
		};
		vkUpdateDescriptorSets(device, static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, NULL);

		// Composition descriptor set
		allocInfo =	vks::initializers::descriptorSetAllocateInfo(descriptorPool, &descriptorSetLayouts.composition, 1);
		VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &descriptorSets.composition));

		colorDescriptors = {
			vks::initializers::descriptorImageInfo(offscreen.sampler, offscreen.color[0].view, vk::ImageLayout::eShaderReadOnlyOptimal),
			vks::initializers::descriptorImageInfo(offscreen.sampler, filterPass.color[0].view, vk::ImageLayout::eShaderReadOnlyOptimal),
		};

		writeDescriptorSets = {
			vks::initializers::writeDescriptorSet(descriptorSets.composition, vk::DescriptorType::eCombinedImageSampler, 0, &colorDescriptors[0]),
			vks::initializers::writeDescriptorSet(descriptorSets.composition, vk::DescriptorType::eCombinedImageSampler, 1, &colorDescriptors[1]),
		};
		vkUpdateDescriptorSets(device, static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, NULL);
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
				vk::CullModeFlagBits::eBack,
				vk::FrontFace::eCounterClockwise,
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
				VK_FALSE,
				VK_FALSE,
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
				static_cast<uint32_t>(dynamicStateEnables.size()),
				0);

		vk::GraphicsPipelineCreateInfo pipelineCreateInfo =
			vks::initializers::pipelineCreateInfo(
				pipelineLayouts.models,
				renderPass,
				0);

		std::vector<vk::PipelineColorBlendAttachmentState> blendAttachmentStates = {
			vks::initializers::pipelineColorBlendAttachmentState(0xf, VK_FALSE),
			vks::initializers::pipelineColorBlendAttachmentState(0xf, VK_FALSE),
		};

		pipelineCreateInfo.pInputAssemblyState = &inputAssemblyState;
		pipelineCreateInfo.pRasterizationState = &rasterizationState;
		pipelineCreateInfo.pColorBlendState = &colorBlendState;
		pipelineCreateInfo.pMultisampleState = &multisampleState;
		pipelineCreateInfo.pViewportState = &viewportState;
		pipelineCreateInfo.pDepthStencilState = &depthStencilState;
		pipelineCreateInfo.pDynamicState = &dynamicState;

		std::array<vk::PipelineShaderStageCreateInfo, 2> shaderStages;
		pipelineCreateInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
		pipelineCreateInfo.pStages = shaderStages.data();

		vk::SpecializationInfo specializationInfo;
		std::array<vk::SpecializationMapEntry, 1> specializationMapEntries;

		// Full screen pipelines

		// Empty vertex input state, full screen triangles are generated by the vertex shader
		vk::PipelineVertexInputStateCreateInfo emptyInputState = vks::initializers::pipelineVertexInputStateCreateInfo();
		pipelineCreateInfo.pVertexInputState = &emptyInputState;

		// Final fullscreen composition pass pipeline
		shaderStages[0] = loadShader(getAssetPath() + "shaders/hdr/composition.vert.spv", vk::ShaderStageFlagBits::eVertex);
		shaderStages[1] = loadShader(getAssetPath() + "shaders/hdr/composition.frag.spv", vk::ShaderStageFlagBits::eFragment);
		pipelineCreateInfo.layout = pipelineLayouts.composition;
		pipelineCreateInfo.renderPass = renderPass;
		rasterizationState.cullMode = vk::CullModeFlagBits::eBack;
		rasterizationState.cullMode = vk::CullModeFlagBits::eNone;
		colorBlendState.attachmentCount = 1;
		colorBlendState.pAttachments = blendAttachmentStates.data();
		VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCreateInfo, nullptr, &pipelines.composition));

		// Bloom pass
		shaderStages[0] = loadShader(getAssetPath() + "shaders/hdr/bloom.vert.spv", vk::ShaderStageFlagBits::eVertex);
		shaderStages[1] = loadShader(getAssetPath() + "shaders/hdr/bloom.frag.spv", vk::ShaderStageFlagBits::eFragment);
		colorBlendState.pAttachments = &blendAttachmentState;
		blendAttachmentState.colorWriteMask = 0xF;
		blendAttachmentState.blendEnable = VK_TRUE;
		blendAttachmentState.colorBlendOp = vk::BlendOp::eAdd;
		blendAttachmentState.srcColorBlendFactor = vk::BlendFactor::eOne;
		blendAttachmentState.dstColorBlendFactor = vk::BlendFactor::eOne;
		blendAttachmentState.alphaBlendOp = vk::BlendOp::eAdd;
		blendAttachmentState.srcAlphaBlendFactor = vk::BlendFactor::eSrcAlpha;
		blendAttachmentState.dstAlphaBlendFactor = vk::BlendFactor::eDstAlpha;

		// Set constant parameters via specialization constants
		specializationMapEntries[0] = vks::initializers::specializationMapEntry(0, 0, sizeof(uint32_t));
		uint32_t dir = 1;
		specializationInfo = vks::initializers::specializationInfo(1, specializationMapEntries.data(), sizeof(dir), &dir);
		shaderStages[1].pSpecializationInfo = &specializationInfo;

		VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCreateInfo, nullptr, &pipelines.bloom[0]));

		// Second blur pass (into separate framebuffer)
		pipelineCreateInfo.renderPass = filterPass.renderPass;
		dir = 0;
		VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCreateInfo, nullptr, &pipelines.bloom[1]));

		// Object rendering pipelines 

		// Vertex bindings an attributes for model rendering
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

		pipelineCreateInfo.renderPass = renderPass;
		pipelineCreateInfo.pVertexInputState = &vertexInputState;

		// Skybox pipeline (background cube)
		blendAttachmentState.blendEnable = VK_FALSE;
		pipelineCreateInfo.layout = pipelineLayouts.models;
		pipelineCreateInfo.renderPass = offscreen.renderPass;
		colorBlendState.attachmentCount = 2;
		colorBlendState.pAttachments = blendAttachmentStates.data();

		shaderStages[0] = loadShader(getAssetPath() + "shaders/hdr/gbuffer.vert.spv", vk::ShaderStageFlagBits::eVertex);
		shaderStages[1] = loadShader(getAssetPath() + "shaders/hdr/gbuffer.frag.spv", vk::ShaderStageFlagBits::eFragment);

		// Set constant parameters via specialization constants
		specializationMapEntries[0] = vks::initializers::specializationMapEntry(0, 0, sizeof(uint32_t));
		uint32_t shadertype = 0;
		specializationInfo = vks::initializers::specializationInfo(1, specializationMapEntries.data(), sizeof(shadertype), &shadertype);
		shaderStages[0].pSpecializationInfo = &specializationInfo;
		shaderStages[1].pSpecializationInfo = &specializationInfo;

		VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCreateInfo, nullptr, &pipelines.skybox));

		// Object rendering pipeline
		shadertype = 1;

		// Enable depth test and write
		depthStencilState.depthWriteEnable = VK_TRUE;
		depthStencilState.depthTestEnable = VK_TRUE;
		// Flip cull mode
		rasterizationState.cullMode = vk::CullModeFlagBits::eNone;
		VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCreateInfo, nullptr, &pipelines.reflect));
	}

	// Prepare and initialize uniform buffer containing shader uniforms
	void prepareUniformBuffers()
	{
		// Matrices vertex shader uniform buffer
		VK_CHECK_RESULT(vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eUniformBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			&uniformBuffers.matrices,
			sizeof(uboVS)));

		// Params
		VK_CHECK_RESULT(vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eUniformBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			&uniformBuffers.params,
			sizeof(uboParams)));

		// Map persistent
		VK_CHECK_RESULT(uniformBuffers.matrices.map());
		VK_CHECK_RESULT(uniformBuffers.params.map());

		updateUniformBuffers();
		updateParams();
	}

	void updateUniformBuffers()
	{
		uboVS.projection = camera.matrices.perspective;
		uboVS.modelview = camera.matrices.view;
		memcpy(uniformBuffers.matrices.mapped, &uboVS, sizeof(uboVS));
	}

	void updateParams()
	{
		memcpy(uniformBuffers.params.mapped, &uboParams, sizeof(uboParams));
	}

	void draw()
	{
		VulkanExampleBase::prepareFrame();

		submitInfo.pWaitSemaphores = &semaphores.presentComplete;
		submitInfo.pSignalSemaphores = &offscreen.semaphore;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &offscreen.cmdBuffer;
		VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));

		submitInfo.pWaitSemaphores = &offscreen.semaphore;
		submitInfo.pSignalSemaphores = &semaphores.renderComplete;
		submitInfo.pCommandBuffers = &drawCmdBuffers[currentBuffer];
		VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));

		VulkanExampleBase::submitFrame();
	}

	void prepare()
	{
		VulkanExampleBase::prepare();
		loadAssets();
		prepareUniformBuffers();
		prepareoffscreenfer();
		setupDescriptorSetLayout();
		preparePipelines();
		setupDescriptorPool();
		setupDescriptorSets();
		buildCommandBuffers();
		buildDeferredCommandBuffer();
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

	void toggleSkyBox()
	{
		displaySkybox = !displaySkybox;
		reBuildCommandBuffers();
	}

	void toggleBloom()
	{
		bloom = !bloom;
		reBuildCommandBuffers();
	}

	void toggleObject()
	{
		models.objectIndex++;
		if (models.objectIndex >= static_cast<uint32_t>(models.objects.size()))
		{
			models.objectIndex = 0;
		}
		reBuildCommandBuffers();
	}

	void changeExposure(float delta)
	{
		uboParams.exposure += delta;
		if (uboParams.exposure < 0.0f) {
			uboParams.exposure = 0.0f;
		}
		updateParams();
		updateTextOverlay();
	}

	virtual void keyPressed(uint32_t keyCode)
	{
		switch (keyCode)
		{
		case KEY_B:
		case GAMEPAD_BUTTON_Y:
			toggleBloom();
			break;
		case KEY_S:
		case GAMEPAD_BUTTON_A:
			toggleSkyBox();
			break;
		case KEY_SPACE:
		case GAMEPAD_BUTTON_X:
			toggleObject();
			break;
		case KEY_KPADD:
		case GAMEPAD_BUTTON_R1:
			changeExposure(0.05f);
			break;
		case KEY_KPSUB:
		case GAMEPAD_BUTTON_L1:
			changeExposure(-0.05f);
			break;
		}
	}

	virtual void getOverlayText(VulkanTextOverlay *textOverlay)
	{
		std::stringstream ss;
		ss << std::setprecision(2) << std::fixed << uboParams.exposure;
#if defined(__ANDROID__)
		textOverlay->addText("Exposure: " + ss.str() + " (L1/R1)", 5.0f, 85.0f, VulkanTextOverlay::alignLeft);
#else
		textOverlay->addText("Exposure: " + ss.str() + " (+/-)", 5.0f, 85.0f, VulkanTextOverlay::alignLeft);
#endif
	}
};

VULKAN_EXAMPLE_MAIN()
