/*
* Vulkan Example - Screen space ambient occlusion example
*
* Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include "VulkanExampleBase.hpp"
#include "VulkanTexture.hpp"
#include "VulkanModel.hpp"

#include <random>

#define VERTEX_BUFFER_BIND_ID 0
#define ENABLE_VALIDATION false

#define SSAO_KERNEL_SIZE 32
#define SSAO_RADIUS 0.5f

#if defined(__ANDROID__)
#define SSAO_NOISE_DIM 8
#else
#define SSAO_NOISE_DIM 4
#endif

class VulkanExample : public VulkanExampleBase
{
public:
	struct {
		vks::Texture2D ssaoNoise;
	} textures;

	// Vertex layout for the models
	vks::VertexLayout vertexLayout = vks::VertexLayout({
		vks::VERTEX_COMPONENT_POSITION,
		vks::VERTEX_COMPONENT_UV,
		vks::VERTEX_COMPONENT_COLOR,
		vks::VERTEX_COMPONENT_NORMAL,
	});

	struct {
		vks::Model scene;
	} models;

	struct {
		vk::PipelineVertexInputStateCreateInfo inputState;
		std::vector<vk::VertexInputBindingDescription> bindingDescriptions;
		std::vector<vk::VertexInputAttributeDescription> attributeDescriptions;
	} vertices;

	struct UBOSceneMatrices {
		glm::mat4 projection;
		glm::mat4 model;
		glm::mat4 view;
	} uboSceneMatrices;

	struct UBOSSAOParams {
		glm::mat4 projection;
		uint32_t ssao = true;
		uint32_t ssaoOnly = false;
		uint32_t ssaoBlur = true;
	} uboSSAOParams;

	struct {
		vk::Pipeline offscreen;
		vk::Pipeline composition;
		vk::Pipeline ssao;
		vk::Pipeline ssaoBlur;
	} pipelines;

	struct {
		vk::PipelineLayout gBuffer;
		vk::PipelineLayout ssao;
		vk::PipelineLayout ssaoBlur;
		vk::PipelineLayout composition;
	} pipelineLayouts;

	struct {
		const uint32_t count = 5;
		vk::DescriptorSet model;
		vk::DescriptorSet floor;
		vk::DescriptorSet ssao;
		vk::DescriptorSet ssaoBlur;
		vk::DescriptorSet composition;
	} descriptorSets;

	struct {
		vk::DescriptorSetLayout gBuffer;
		vk::DescriptorSetLayout ssao;
		vk::DescriptorSetLayout ssaoBlur;
		vk::DescriptorSetLayout composition;
	} descriptorSetLayouts;

	struct {
		vks::Buffer sceneMatrices;
		vks::Buffer ssaoKernel;
		vks::Buffer ssaoParams;
	} uniformBuffers;

	// Framebuffer for offscreen rendering
	struct FrameBufferAttachment {
		vk::Image image;
		vk::DeviceMemory mem;
		vk::ImageView view;
		vk::Format format;
		void destroy(vk::Device device)
		{
			device.destroyImage(image);
			device.destroyImageView(view);
			device.freeMemory(mem);
		}
	};
	struct FrameBuffer {
		uint32_t width, height;
		vk::Framebuffer frameBuffer;		
		vk::RenderPass renderPass;
		void setSize(int32_t w, int32_t h)
		{
			this->width = w;
			this->height = h;
		}
		void destroy(vk::Device device)
		{
			device.destroyFramebuffer(frameBuffer);
			device.destroyRenderPass(renderPass);
		}
	};

	struct {
		struct Offscreen : public FrameBuffer {
			FrameBufferAttachment position, normal, albedo, depth;
		} offscreen;
		struct SSAO : public FrameBuffer {
			FrameBufferAttachment color;
		} ssao, ssaoBlur;
	} frameBuffers;

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
		title = "Vulkan Example - Screen space ambient occlusion";
		camera.type = Camera::CameraType::firstperson;
		camera.movementSpeed = 5.0f;
#ifndef __ANDROID__
		camera.rotationSpeed = 0.25f;
#endif
		camera.position = { 7.5f, -6.75f, 0.0f };
		camera.setRotation(glm::vec3(5.0f, 90.0f, 0.0f));
		camera.setPerspective(60.0f, (float)width / (float)height, 0.1f, 64.0f);
	}

	~VulkanExample()
	{
		device.destroySampler(colorSampler);

		// Attachments
		frameBuffers.offscreen.position.destroy(device);
		frameBuffers.offscreen.normal.destroy(device);
		frameBuffers.offscreen.albedo.destroy(device);
		frameBuffers.offscreen.depth.destroy(device);
		frameBuffers.ssao.color.destroy(device);
		frameBuffers.ssaoBlur.color.destroy(device);

		// Framebuffers
		frameBuffers.offscreen.destroy(device);
		frameBuffers.ssao.destroy(device);
		frameBuffers.ssaoBlur.destroy(device);

		device.destroyPipeline(pipelines.offscreen);
		device.destroyPipeline(pipelines.composition);
		device.destroyPipeline(pipelines.ssao);
		device.destroyPipeline(pipelines.ssaoBlur);

		device.destroyPipelineLayout(pipelineLayouts.gBuffer);
		device.destroyPipelineLayout(pipelineLayouts.ssao);
		device.destroyPipelineLayout(pipelineLayouts.ssaoBlur);
		device.destroyPipelineLayout(pipelineLayouts.composition);

		device.destroyDescriptorSetLayout(descriptorSetLayouts.gBuffer);
		device.destroyDescriptorSetLayout(descriptorSetLayouts.ssao);
		device.destroyDescriptorSetLayout(descriptorSetLayouts.ssaoBlur);
		device.destroyDescriptorSetLayout(descriptorSetLayouts.composition);

		// Meshes
		models.scene.destroy();

		// Uniform buffers
		uniformBuffers.sceneMatrices.destroy();
		uniformBuffers.ssaoKernel.destroy();
		uniformBuffers.ssaoParams.destroy();

		// Misc
		device.freeCommandBuffers(cmdPool, offScreenCmdBuffer);
		device.destroySemaphore(offscreenSemaphore);
		textures.ssaoNoise.destroy();
	}

	// Create a frame buffer attachment
	void createAttachment(
		vk::Format format,  
		vk::ImageUsageFlags usage,
		FrameBufferAttachment *attachment,
		uint32_t width,
		uint32_t height)
	{
		vk::ImageAspectFlags aspectMask;
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

		assert(aspectMask);

		vk::ImageCreateInfo image = vks::initializers::imageCreateInfo();
		image.imageType = vk::ImageType::e2D;
		image.format = format;
		image.extent = vk::Extent3D{ width, height, 1 };
		image.mipLevels = 1;
		image.arrayLayers = 1;
		image.samples = vk::SampleCountFlagBits::e1;
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

	void prepareOffscreenFramebuffers()
	{
		// Attachments
		vk::CommandBuffer layoutCmd = VulkanExampleBase::createCommandBuffer(vk::CommandBufferLevel::ePrimary, true);

#if defined(__ANDROID__)
		const uint32_t ssaoWidth = width / 2;
		const uint32_t ssaoHeight = height / 2;
#else
		const uint32_t ssaoWidth = width;
		const uint32_t ssaoHeight = height;
#endif

		frameBuffers.offscreen.setSize(width, height);
		frameBuffers.ssao.setSize(ssaoWidth, ssaoHeight);
		frameBuffers.ssaoBlur.setSize(width, height);

		// Find a suitable depth format
		vk::Format attDepthFormat;
		vk::Bool32 validDepthFormat = vks::tools::getSupportedDepthFormat(physicalDevice, &attDepthFormat);
		assert(validDepthFormat);

		// G-Buffer 
		createAttachment(vk::Format::eR32G32B32A32Sfloat, vk::ImageUsageFlagBits::eColorAttachment, &frameBuffers.offscreen.position, width, height);	// Position + Depth
		createAttachment(vk::Format::eR8G8B8A8Unorm, vk::ImageUsageFlagBits::eColorAttachment, &frameBuffers.offscreen.normal, width, height);			// Normals
		createAttachment(vk::Format::eR8G8B8A8Unorm, vk::ImageUsageFlagBits::eColorAttachment, &frameBuffers.offscreen.albedo, width, height);			// Albedo (color)
		createAttachment(attDepthFormat, vk::ImageUsageFlagBits::eDepthStencilAttachment, &frameBuffers.offscreen.depth, width, height);			// Depth

		// SSAO
		createAttachment(vk::Format::eR8Unorm, vk::ImageUsageFlagBits::eColorAttachment, &frameBuffers.ssao.color, ssaoWidth, ssaoHeight);				// Color

		// SSAO blur
		createAttachment(vk::Format::eR8Unorm, vk::ImageUsageFlagBits::eColorAttachment, &frameBuffers.ssaoBlur.color, width, height);					// Color

		VulkanExampleBase::flushCommandBuffer(layoutCmd, queue, true);

		// Render passes

		// G-Buffer creation
		{
			std::array<vk::AttachmentDescription, 4> attachmentDescs = {};

			// Init attachment properties
			for (uint32_t i = 0; i < static_cast<uint32_t>(attachmentDescs.size()); i++)
			{
				attachmentDescs[i].samples = vk::SampleCountFlagBits::e1;
				attachmentDescs[i].loadOp = vk::AttachmentLoadOp::eClear;
				attachmentDescs[i].storeOp = vk::AttachmentStoreOp::eStore;
				attachmentDescs[i].stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
				attachmentDescs[i].stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
				attachmentDescs[i].finalLayout = (i == 3) ? vk::ImageLayout::eDepthStencilReadOnlyOptimal : vk::ImageLayout::eShaderReadOnlyOptimal;
			}

			// Formats
			attachmentDescs[0].format = frameBuffers.offscreen.position.format;
			attachmentDescs[1].format = frameBuffers.offscreen.normal.format;
			attachmentDescs[2].format = frameBuffers.offscreen.albedo.format;
			attachmentDescs[3].format = frameBuffers.offscreen.depth.format;

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

			// Use subpass dependencies for attachment layout transitions
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
			frameBuffers.offscreen.renderPass = device.createRenderPass(renderPassInfo);

			std::array<vk::ImageView, 4> attachments;
			attachments[0] = frameBuffers.offscreen.position.view;
			attachments[1] = frameBuffers.offscreen.normal.view;
			attachments[2] = frameBuffers.offscreen.albedo.view;
			attachments[3] = frameBuffers.offscreen.depth.view;

			vk::FramebufferCreateInfo fbufCreateInfo = vks::initializers::framebufferCreateInfo();
			fbufCreateInfo.renderPass = frameBuffers.offscreen.renderPass;
			fbufCreateInfo.pAttachments = attachments.data();
			fbufCreateInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
			fbufCreateInfo.width = frameBuffers.offscreen.width;
			fbufCreateInfo.height = frameBuffers.offscreen.height;
			fbufCreateInfo.layers = 1;
			frameBuffers.offscreen.frameBuffer = device.createFramebuffer(fbufCreateInfo);
		}

		// SSAO 
		{
			vk::AttachmentDescription attachmentDescription{};
			attachmentDescription.format = frameBuffers.ssao.color.format;
			attachmentDescription.samples = vk::SampleCountFlagBits::e1;
			attachmentDescription.loadOp = vk::AttachmentLoadOp::eClear;
			attachmentDescription.storeOp = vk::AttachmentStoreOp::eStore;
			attachmentDescription.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
			attachmentDescription.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
			attachmentDescription.initialLayout = vk::ImageLayout::eUndefined;
			attachmentDescription.finalLayout = vk::ImageLayout::eShaderReadOnlyOptimal;

			vk::AttachmentReference colorReference = { 0, vk::ImageLayout::eColorAttachmentOptimal };

			vk::SubpassDescription subpass = {};
			subpass.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
			subpass.pColorAttachments = &colorReference;
			subpass.colorAttachmentCount = 1;

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

			renderPassInfo.pAttachments = &attachmentDescription;
			renderPassInfo.attachmentCount = 1;
			renderPassInfo.subpassCount = 1;
			renderPassInfo.pSubpasses = &subpass;
			renderPassInfo.dependencyCount = 2;
			renderPassInfo.pDependencies = dependencies.data();
			frameBuffers.ssao.renderPass = device.createRenderPass(renderPassInfo);

			vk::FramebufferCreateInfo fbufCreateInfo = vks::initializers::framebufferCreateInfo();
			fbufCreateInfo.renderPass = frameBuffers.ssao.renderPass;
			fbufCreateInfo.pAttachments = &frameBuffers.ssao.color.view;
			fbufCreateInfo.attachmentCount = 1;
			fbufCreateInfo.width = frameBuffers.ssao.width;
			fbufCreateInfo.height = frameBuffers.ssao.height;
			fbufCreateInfo.layers = 1;
			frameBuffers.ssao.frameBuffer = device.createFramebuffer(fbufCreateInfo);
		}

		// SSAO Blur 
		{
			vk::AttachmentDescription attachmentDescription{};
			attachmentDescription.format = frameBuffers.ssao.color.format;
			attachmentDescription.samples = vk::SampleCountFlagBits::e1;
			attachmentDescription.loadOp = vk::AttachmentLoadOp::eClear;
			attachmentDescription.storeOp = vk::AttachmentStoreOp::eStore;
			attachmentDescription.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
			attachmentDescription.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
			attachmentDescription.initialLayout = vk::ImageLayout::eUndefined;
			attachmentDescription.finalLayout = vk::ImageLayout::eShaderReadOnlyOptimal;

			vk::AttachmentReference colorReference = { 0, vk::ImageLayout::eColorAttachmentOptimal };

			vk::SubpassDescription subpass = {};
			subpass.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
			subpass.pColorAttachments = &colorReference;
			subpass.colorAttachmentCount = 1;

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

			renderPassInfo.pAttachments = &attachmentDescription;
			renderPassInfo.attachmentCount = 1;
			renderPassInfo.subpassCount = 1;
			renderPassInfo.pSubpasses = &subpass;
			renderPassInfo.dependencyCount = 2;
			renderPassInfo.pDependencies = dependencies.data();
			frameBuffers.ssaoBlur.renderPass = device.createRenderPass(renderPassInfo);

			vk::FramebufferCreateInfo fbufCreateInfo = vks::initializers::framebufferCreateInfo();
			fbufCreateInfo.renderPass = frameBuffers.ssaoBlur.renderPass;
			fbufCreateInfo.pAttachments = &frameBuffers.ssaoBlur.color.view;
			fbufCreateInfo.attachmentCount = 1;
			fbufCreateInfo.width = frameBuffers.ssaoBlur.width;
			fbufCreateInfo.height = frameBuffers.ssaoBlur.height;
			fbufCreateInfo.layers = 1;
			frameBuffers.ssaoBlur.frameBuffer = device.createFramebuffer(fbufCreateInfo);
		}

		// Shared sampler used for all color attachments
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
		std::vector<vk::DeviceSize> offsets = { 0 };

		if (!offScreenCmdBuffer)
		{
			offScreenCmdBuffer = VulkanExampleBase::createCommandBuffer(vk::CommandBufferLevel::ePrimary, false);
		}

		// Create a semaphore used to synchronize offscreen rendering and usage
		vk::SemaphoreCreateInfo semaphoreCreateInfo = vks::initializers::semaphoreCreateInfo();
		offscreenSemaphore = device.createSemaphore(semaphoreCreateInfo);

		vk::CommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();

		// Clear values for all attachments written in the fragment sahder
		std::vector<vk::ClearValue> clearValues(4);
		clearValues[0].color = vk::ClearColorValue{ std::array<float, 4>{ 0.0f, 0.0f, 0.0f, 1.0f } };
		clearValues[1].color = vk::ClearColorValue{ std::array<float, 4>{ 0.0f, 0.0f, 0.0f, 1.0f } };
		clearValues[2].color = vk::ClearColorValue{ std::array<float, 4>{ 0.0f, 0.0f, 0.0f, 1.0f } };
		clearValues[3].depthStencil = vk::ClearDepthStencilValue{ 1.0f, 0 };

		vk::RenderPassBeginInfo renderPassBeginInfo = vks::initializers::renderPassBeginInfo();
		renderPassBeginInfo.renderPass =  frameBuffers.offscreen.renderPass;
		renderPassBeginInfo.framebuffer = frameBuffers.offscreen.frameBuffer;
		renderPassBeginInfo.renderArea.extent.width = frameBuffers.offscreen.width;
		renderPassBeginInfo.renderArea.extent.height = frameBuffers.offscreen.height;
		renderPassBeginInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
		renderPassBeginInfo.pClearValues = clearValues.data();

		offScreenCmdBuffer.begin(cmdBufInfo);

		// First pass: Fill G-Buffer components (positions+depth, normals, albedo) using MRT
		// -------------------------------------------------------------------------------------------------------

		offScreenCmdBuffer.beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);

		vk::Viewport viewport = vks::initializers::viewport((float)frameBuffers.offscreen.width, (float)frameBuffers.offscreen.height, 0.0f, 1.0f);
		offScreenCmdBuffer.setViewport(0, viewport);

		vk::Rect2D scissor = vks::initializers::rect2D(frameBuffers.offscreen.width, frameBuffers.offscreen.height, 0, 0);
		offScreenCmdBuffer.setScissor(0, scissor);

		offScreenCmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.offscreen);

		offScreenCmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayouts.gBuffer, 0, descriptorSets.floor, nullptr);
		offScreenCmdBuffer.bindVertexBuffers(VERTEX_BUFFER_BIND_ID, models.scene.vertices.buffer, offsets);
		offScreenCmdBuffer.bindIndexBuffer(models.scene.indices.buffer, 0, vk::IndexType::eUint32);
		offScreenCmdBuffer.drawIndexed(models.scene.indexCount, 1, 0, 0, 0);

		offScreenCmdBuffer.endRenderPass();

		// Second pass: SSAO generation
		// -------------------------------------------------------------------------------------------------------

		clearValues[0].color = vk::ClearColorValue{ std::array<float, 4>{ 0.0f, 0.0f, 0.0f, 1.0f } };
		clearValues[1].depthStencil = vk::ClearDepthStencilValue{ 1.0f, 0 };

		renderPassBeginInfo.framebuffer = frameBuffers.ssao.frameBuffer;
		renderPassBeginInfo.renderPass = frameBuffers.ssao.renderPass;
		renderPassBeginInfo.renderArea.extent.width = frameBuffers.ssao.width;
		renderPassBeginInfo.renderArea.extent.height = frameBuffers.ssao.height;
		renderPassBeginInfo.clearValueCount = 2;
		renderPassBeginInfo.pClearValues = clearValues.data();

		offScreenCmdBuffer.beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);

		viewport = vks::initializers::viewport((float)frameBuffers.ssao.width, (float)frameBuffers.ssao.height, 0.0f, 1.0f);
		offScreenCmdBuffer.setViewport(0, viewport);
		scissor = vks::initializers::rect2D(frameBuffers.ssao.width, frameBuffers.ssao.height, 0, 0);
		offScreenCmdBuffer.setScissor(0, scissor);

		offScreenCmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayouts.ssao, 0, descriptorSets.ssao, nullptr);
		offScreenCmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.ssao);
		vkCmdDraw(offScreenCmdBuffer, 3, 1, 0, 0);

		offScreenCmdBuffer.endRenderPass();

		// Third pass: SSAO blur
		// -------------------------------------------------------------------------------------------------------

		renderPassBeginInfo.framebuffer = frameBuffers.ssaoBlur.frameBuffer;
		renderPassBeginInfo.renderPass = frameBuffers.ssaoBlur.renderPass;
		renderPassBeginInfo.renderArea.extent.width = frameBuffers.ssaoBlur.width;
		renderPassBeginInfo.renderArea.extent.height = frameBuffers.ssaoBlur.height;

		offScreenCmdBuffer.beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);

		viewport = vks::initializers::viewport((float)frameBuffers.ssaoBlur.width, (float)frameBuffers.ssaoBlur.height, 0.0f, 1.0f);
		offScreenCmdBuffer.setViewport(0, viewport);
		scissor = vks::initializers::rect2D(frameBuffers.ssaoBlur.width, frameBuffers.ssaoBlur.height, 0, 0);
		offScreenCmdBuffer.setScissor(0, scissor);

		offScreenCmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayouts.ssaoBlur, 0, descriptorSets.ssaoBlur, nullptr);
		offScreenCmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.ssaoBlur);
		vkCmdDraw(offScreenCmdBuffer, 3, 1, 0, 0);

		offScreenCmdBuffer.endRenderPass();

		offScreenCmdBuffer.end();
	}

	void loadAssets()
	{
		vks::ModelCreateInfo modelCreateInfo;
		modelCreateInfo.scale = glm::vec3(0.5f);
		modelCreateInfo.uvscale = glm::vec2(1.0f);
		modelCreateInfo.center = glm::vec3(0.0f, 0.0f, 0.0f);
		models.scene.loadFromFile(getAssetPath() + "models/sibenik/sibenik.dae", vertexLayout, &modelCreateInfo, vulkanDevice, queue);
	}

	void buildCommandBuffers()
	{
		vk::CommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();

		vk::ClearValue clearValues[2];
		clearValues[0].color = vk::ClearColorValue{ std::array<float, 4>{ 0.0f, 0.0f, 0.0f, 0.0f } };
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

			//std::vector<vk::DeviceSize> offsets = { 0 };
			drawCmdBuffers[i].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayouts.composition, 0, descriptorSets.composition, nullptr);

			// Final composition pass
			drawCmdBuffers[i].bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.composition);
			vkCmdDraw(drawCmdBuffers[i], 3, 1, 0, 0);

			drawCmdBuffers[i].endRenderPass();

			drawCmdBuffers[i].end();
		}
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
			vks::initializers::descriptorPoolSize(vk::DescriptorType::eUniformBuffer, 10),
			vks::initializers::descriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, 12)
		};

		vk::DescriptorPoolCreateInfo descriptorPoolInfo =
			vks::initializers::descriptorPoolCreateInfo(
				static_cast<uint32_t>(poolSizes.size()),
				poolSizes.data(),
				descriptorSets.count);

		descriptorPool = device.createDescriptorPool(descriptorPoolInfo);
	}

	void setupLayoutsAndDescriptors()
	{
		std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings;
		vk::DescriptorSetLayoutCreateInfo setLayoutCreateInfo;
		vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo = vks::initializers::pipelineLayoutCreateInfo();
		vk::DescriptorSetAllocateInfo descriptorAllocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorPool, nullptr, 1);
		std::vector<vk::WriteDescriptorSet> writeDescriptorSets;
		std::vector<vk::DescriptorImageInfo> imageDescriptors;

		// G-Buffer creation (offscreen scene rendering)
		setLayoutBindings = {
			vks::initializers::descriptorSetLayoutBinding(vk::DescriptorType::eUniformBuffer, vk::ShaderStageFlagBits::eVertex, 0),								// VS UBO
			vks::initializers::descriptorSetLayoutBinding(vk::DescriptorType::eCombinedImageSampler, vk::ShaderStageFlagBits::eFragment, 1),						// FS Color
		};
		setLayoutCreateInfo = vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
		descriptorSetLayouts.gBuffer = device.createDescriptorSetLayout(setLayoutCreateInfo);
		pipelineLayoutCreateInfo.pSetLayouts = &descriptorSetLayouts.gBuffer;
		pipelineLayouts.gBuffer = device.createPipelineLayout(pipelineLayoutCreateInfo);
		descriptorAllocInfo.pSetLayouts = &descriptorSetLayouts.gBuffer;
		descriptorSets.floor = device.allocateDescriptorSets(descriptorAllocInfo)[0];
		writeDescriptorSets = {
			vks::initializers::writeDescriptorSet(descriptorSets.floor, vk::DescriptorType::eUniformBuffer, 0, &uniformBuffers.sceneMatrices.descriptor),
		};
		device.updateDescriptorSets(writeDescriptorSets, nullptr);

		// SSAO Generation
		setLayoutBindings = {
			vks::initializers::descriptorSetLayoutBinding(vk::DescriptorType::eCombinedImageSampler, vk::ShaderStageFlagBits::eFragment, 0),						// FS Position+Depth
			vks::initializers::descriptorSetLayoutBinding(vk::DescriptorType::eCombinedImageSampler, vk::ShaderStageFlagBits::eFragment, 1),						// FS Normals
			vks::initializers::descriptorSetLayoutBinding(vk::DescriptorType::eCombinedImageSampler, vk::ShaderStageFlagBits::eFragment, 2),						// FS SSAO Noise
			vks::initializers::descriptorSetLayoutBinding(vk::DescriptorType::eUniformBuffer, vk::ShaderStageFlagBits::eFragment, 3),								// FS SSAO Kernel UBO
			vks::initializers::descriptorSetLayoutBinding(vk::DescriptorType::eUniformBuffer, vk::ShaderStageFlagBits::eFragment, 4),								// FS Params UBO 
		};
		setLayoutCreateInfo = vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
		descriptorSetLayouts.ssao = device.createDescriptorSetLayout(setLayoutCreateInfo);
		pipelineLayoutCreateInfo.pSetLayouts = &descriptorSetLayouts.ssao;
		pipelineLayouts.ssao = device.createPipelineLayout(pipelineLayoutCreateInfo);
		descriptorAllocInfo.pSetLayouts = &descriptorSetLayouts.ssao;
		descriptorSets.ssao = device.allocateDescriptorSets(descriptorAllocInfo)[0];
		imageDescriptors = {
			vks::initializers::descriptorImageInfo(colorSampler, frameBuffers.offscreen.position.view, vk::ImageLayout::eShaderReadOnlyOptimal),
			vks::initializers::descriptorImageInfo(colorSampler, frameBuffers.offscreen.normal.view, vk::ImageLayout::eShaderReadOnlyOptimal),
		};
		writeDescriptorSets = {
			vks::initializers::writeDescriptorSet(descriptorSets.ssao, vk::DescriptorType::eCombinedImageSampler, 0, &imageDescriptors[0]),					// FS Position+Depth
			vks::initializers::writeDescriptorSet(descriptorSets.ssao, vk::DescriptorType::eCombinedImageSampler, 1, &imageDescriptors[1]),					// FS Normals
			vks::initializers::writeDescriptorSet(descriptorSets.ssao, vk::DescriptorType::eCombinedImageSampler, 2, &textures.ssaoNoise.descriptor),		// FS SSAO Noise
			vks::initializers::writeDescriptorSet(descriptorSets.ssao, vk::DescriptorType::eUniformBuffer, 3, &uniformBuffers.ssaoKernel.descriptor),		// FS SSAO Kernel UBO
			vks::initializers::writeDescriptorSet(descriptorSets.ssao, vk::DescriptorType::eUniformBuffer, 4, &uniformBuffers.ssaoParams.descriptor),		// FS SSAO Params UBO
		};
		device.updateDescriptorSets(writeDescriptorSets, nullptr);

		// SSAO Blur
		setLayoutBindings = {
			vks::initializers::descriptorSetLayoutBinding(vk::DescriptorType::eCombinedImageSampler, vk::ShaderStageFlagBits::eFragment, 0),						// FS Sampler SSAO
		};
		setLayoutCreateInfo = vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
		descriptorSetLayouts.ssaoBlur = device.createDescriptorSetLayout(setLayoutCreateInfo);
		pipelineLayoutCreateInfo.pSetLayouts = &descriptorSetLayouts.ssaoBlur;
		pipelineLayouts.ssaoBlur = device.createPipelineLayout(pipelineLayoutCreateInfo);
		descriptorAllocInfo.pSetLayouts = &descriptorSetLayouts.ssaoBlur;
		descriptorSets.ssaoBlur = device.allocateDescriptorSets(descriptorAllocInfo)[0];
		imageDescriptors = {
			vks::initializers::descriptorImageInfo(colorSampler, frameBuffers.ssao.color.view, vk::ImageLayout::eShaderReadOnlyOptimal),
		};
		writeDescriptorSets = {
			vks::initializers::writeDescriptorSet(descriptorSets.ssaoBlur, vk::DescriptorType::eCombinedImageSampler, 0, &imageDescriptors[0]),
		};
		device.updateDescriptorSets(writeDescriptorSets, nullptr);

		// Composition
		setLayoutBindings = {
			vks::initializers::descriptorSetLayoutBinding(vk::DescriptorType::eCombinedImageSampler, vk::ShaderStageFlagBits::eFragment, 0),						// FS Position+Depth
			vks::initializers::descriptorSetLayoutBinding(vk::DescriptorType::eCombinedImageSampler, vk::ShaderStageFlagBits::eFragment, 1),						// FS Normals
			vks::initializers::descriptorSetLayoutBinding(vk::DescriptorType::eCombinedImageSampler, vk::ShaderStageFlagBits::eFragment, 2),						// FS Albedo
			vks::initializers::descriptorSetLayoutBinding(vk::DescriptorType::eCombinedImageSampler, vk::ShaderStageFlagBits::eFragment, 3),						// FS SSAO
			vks::initializers::descriptorSetLayoutBinding(vk::DescriptorType::eCombinedImageSampler, vk::ShaderStageFlagBits::eFragment, 4),						// FS SSAO blurred
			vks::initializers::descriptorSetLayoutBinding(vk::DescriptorType::eUniformBuffer, vk::ShaderStageFlagBits::eFragment, 5),								// FS Lights UBO 
		};
		setLayoutCreateInfo = vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
		descriptorSetLayouts.composition = device.createDescriptorSetLayout(setLayoutCreateInfo);
		pipelineLayoutCreateInfo.pSetLayouts = &descriptorSetLayouts.composition;
		pipelineLayouts.composition = device.createPipelineLayout(pipelineLayoutCreateInfo);
		descriptorAllocInfo.pSetLayouts = &descriptorSetLayouts.composition;
		descriptorSets.composition = device.allocateDescriptorSets(descriptorAllocInfo)[0];
		imageDescriptors = {
			vks::initializers::descriptorImageInfo(colorSampler, frameBuffers.offscreen.position.view, vk::ImageLayout::eShaderReadOnlyOptimal),
			vks::initializers::descriptorImageInfo(colorSampler, frameBuffers.offscreen.normal.view, vk::ImageLayout::eShaderReadOnlyOptimal),
			vks::initializers::descriptorImageInfo(colorSampler, frameBuffers.offscreen.albedo.view, vk::ImageLayout::eShaderReadOnlyOptimal),
			vks::initializers::descriptorImageInfo(colorSampler, frameBuffers.ssao.color.view, vk::ImageLayout::eShaderReadOnlyOptimal),  
			vks::initializers::descriptorImageInfo(colorSampler, frameBuffers.ssaoBlur.color.view, vk::ImageLayout::eShaderReadOnlyOptimal), 
		};
		writeDescriptorSets = {
			vks::initializers::writeDescriptorSet(descriptorSets.composition, vk::DescriptorType::eCombinedImageSampler, 0, &imageDescriptors[0]),			// FS Sampler Position+Depth
			vks::initializers::writeDescriptorSet(descriptorSets.composition, vk::DescriptorType::eCombinedImageSampler, 1, &imageDescriptors[1]),			// FS Sampler Normals
			vks::initializers::writeDescriptorSet(descriptorSets.composition, vk::DescriptorType::eCombinedImageSampler, 2, &imageDescriptors[2]),			// FS Sampler Albedo
			vks::initializers::writeDescriptorSet(descriptorSets.composition, vk::DescriptorType::eCombinedImageSampler, 3, &imageDescriptors[3]),			// FS Sampler SSAO 
			vks::initializers::writeDescriptorSet(descriptorSets.composition, vk::DescriptorType::eCombinedImageSampler, 4, &imageDescriptors[4]),			// FS Sampler SSAO blurred
			vks::initializers::writeDescriptorSet(descriptorSets.composition, vk::DescriptorType::eUniformBuffer, 5, &uniformBuffers.ssaoParams.descriptor),	// FS SSAO Params UBO
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

		// Final composition pass pipeline
		std::array<vk::PipelineShaderStageCreateInfo, 2> shaderStages;

		shaderStages[0] = loadShader(getAssetPath() + "shaders/ssao/fullscreen.vert.spv", vk::ShaderStageFlagBits::eVertex);
		shaderStages[1] = loadShader(getAssetPath() + "shaders/ssao/composition.frag.spv", vk::ShaderStageFlagBits::eFragment);

		vk::GraphicsPipelineCreateInfo pipelineCreateInfo =
			vks::initializers::pipelineCreateInfo(
				pipelineLayouts.composition,
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

		vk::PipelineVertexInputStateCreateInfo emptyInputState{};

		emptyInputState.vertexAttributeDescriptionCount = 0;
		emptyInputState.pVertexAttributeDescriptions = nullptr;
		emptyInputState.vertexBindingDescriptionCount = 0;
		emptyInputState.pVertexBindingDescriptions = nullptr;
		pipelineCreateInfo.pVertexInputState = &emptyInputState;

		pipelines.composition = device.createGraphicsPipelines(pipelineCache, pipelineCreateInfo)[0];

		pipelineCreateInfo.pVertexInputState = &emptyInputState;

		// SSAO Pass
		shaderStages[0] = loadShader(getAssetPath() + "shaders/ssao/fullscreen.vert.spv", vk::ShaderStageFlagBits::eVertex);
		shaderStages[1] = loadShader(getAssetPath() + "shaders/ssao/ssao.frag.spv", vk::ShaderStageFlagBits::eFragment);
		{
			// Set constant parameters via specialization constants
			std::array<vk::SpecializationMapEntry, 2> specializationMapEntries;
			specializationMapEntries[0] = vks::initializers::specializationMapEntry(0, 0, sizeof(uint32_t));				// SSAO Kernel size
			specializationMapEntries[1] = vks::initializers::specializationMapEntry(1, sizeof(uint32_t), sizeof(float));	// SSAO radius
			struct {
				uint32_t kernelSize = SSAO_KERNEL_SIZE;
				float radius = SSAO_RADIUS;
			} specializationData;
			vk::SpecializationInfo specializationInfo = vks::initializers::specializationInfo(2, specializationMapEntries.data(), sizeof(specializationData), &specializationData);
			shaderStages[1].pSpecializationInfo = &specializationInfo;
			pipelineCreateInfo.renderPass = frameBuffers.ssao.renderPass;
			pipelineCreateInfo.layout = pipelineLayouts.ssao;
			pipelines.ssao = device.createGraphicsPipelines(pipelineCache, pipelineCreateInfo)[0];
		}


		// SSAO blur pass
		shaderStages[0] = loadShader(getAssetPath() + "shaders/ssao/fullscreen.vert.spv", vk::ShaderStageFlagBits::eVertex);
		shaderStages[1] = loadShader(getAssetPath() + "shaders/ssao/blur.frag.spv", vk::ShaderStageFlagBits::eFragment);
		pipelineCreateInfo.renderPass = frameBuffers.ssaoBlur.renderPass;
		pipelineCreateInfo.layout = pipelineLayouts.ssaoBlur;
		pipelines.ssaoBlur = device.createGraphicsPipelines(pipelineCache, pipelineCreateInfo)[0];

		// Fill G-Buffer
		shaderStages[0] = loadShader(getAssetPath() + "shaders/ssao/gbuffer.vert.spv", vk::ShaderStageFlagBits::eVertex);
		shaderStages[1] = loadShader(getAssetPath() + "shaders/ssao/gbuffer.frag.spv", vk::ShaderStageFlagBits::eFragment);
		pipelineCreateInfo.pVertexInputState = &vertices.inputState;
		pipelineCreateInfo.renderPass = frameBuffers.offscreen.renderPass;
		pipelineCreateInfo.layout = pipelineLayouts.gBuffer;
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
	}

	float lerp(float a, float b, float f)
	{
		return a + f * (b - a);
	}

	// Prepare and initialize uniform buffer containing shader uniforms
	void prepareUniformBuffers()
	{
		// Scene matrices
		vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eUniformBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			&uniformBuffers.sceneMatrices,
			sizeof(uboSceneMatrices));

		// SSAO parameters 
		vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eUniformBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			&uniformBuffers.ssaoParams,
			sizeof(uboSSAOParams));

		// Update
		updateUniformBufferMatrices();
		updateUniformBufferSSAOParams();

		// SSAO
		std::uniform_real_distribution<float> rndDist(0.0f, 1.0f);
		std::random_device rndDev;
		std::default_random_engine rndGen;

		// Sample kernel
		std::vector<glm::vec4> ssaoKernel(SSAO_KERNEL_SIZE);
		for (uint32_t i = 0; i < SSAO_KERNEL_SIZE; ++i)
		{
			glm::vec3 sample(rndDist(rndGen) * 2.0 - 1.0, rndDist(rndGen) * 2.0 - 1.0, rndDist(rndGen));
			sample = glm::normalize(sample);
			sample *= rndDist(rndGen);
			float scale = float(i) / float(SSAO_KERNEL_SIZE);
			scale = lerp(0.1f, 1.0f, scale * scale);
			ssaoKernel[i] = glm::vec4(sample * scale, 0.0f);
		}

		// Upload as UBO
		vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eUniformBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			&uniformBuffers.ssaoKernel,
			ssaoKernel.size() * sizeof(glm::vec4),
			ssaoKernel.data());

		// Random noise
		std::vector<glm::vec4> ssaoNoise(SSAO_NOISE_DIM * SSAO_NOISE_DIM);
		for (uint32_t i = 0; i < static_cast<uint32_t>(ssaoNoise.size()); i++)
		{
			ssaoNoise[i] = glm::vec4(rndDist(rndGen) * 2.0f - 1.0f, rndDist(rndGen) * 2.0f - 1.0f, 0.0f, 0.0f);
		}
		// Upload as texture
		textures.ssaoNoise.fromBuffer(ssaoNoise.data(), ssaoNoise.size() * sizeof(glm::vec4), vk::Format::eR32G32B32A32Sfloat, SSAO_NOISE_DIM, SSAO_NOISE_DIM, vulkanDevice, queue, vk::Filter::eNearest);
	}

	void updateUniformBufferMatrices()
	{
		uboSceneMatrices.projection = camera.matrices.perspective;
		uboSceneMatrices.view = camera.matrices.view;
		uboSceneMatrices.model = glm::mat4();

		uniformBuffers.sceneMatrices.map();
		uniformBuffers.sceneMatrices.copyTo(&uboSceneMatrices, sizeof(uboSceneMatrices));
		uniformBuffers.sceneMatrices.unmap();
	}

	void updateUniformBufferSSAOParams()
	{
		uboSSAOParams.projection = camera.matrices.perspective;

		uniformBuffers.ssaoParams.map();
		uniformBuffers.ssaoParams.copyTo(&uboSSAOParams, sizeof(uboSSAOParams));
		uniformBuffers.ssaoParams.unmap();
	}

	void draw()
	{
		VulkanExampleBase::prepareFrame();

		// Offscreen rendering
		submitInfo.pWaitSemaphores = &semaphores.presentComplete;
		submitInfo.pSignalSemaphores = &offscreenSemaphore;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &offScreenCmdBuffer;
		queue.submit(submitInfo, vk::Fence(nullptr));

		// Scene rendering
		submitInfo.pWaitSemaphores = &offscreenSemaphore;
		submitInfo.pSignalSemaphores = &semaphores.renderComplete;
		submitInfo.pCommandBuffers = &drawCmdBuffers[currentBuffer];
		queue.submit(submitInfo, vk::Fence(nullptr));

		VulkanExampleBase::submitFrame();
	}

	void prepare()
	{
		VulkanExampleBase::prepare();
		loadAssets();
		setupVertexDescriptions();
		prepareOffscreenFramebuffers();
		prepareUniformBuffers();
		setupDescriptorPool();
		setupLayoutsAndDescriptors();
		preparePipelines();
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
		updateUniformBufferMatrices();
		updateUniformBufferSSAOParams();
	}

	void toggleSSAO()
	{
		uboSSAOParams.ssao = !uboSSAOParams.ssao;
		updateUniformBufferSSAOParams();
	}

	void toggleSSAOBlur()
	{
		uboSSAOParams.ssaoBlur = !uboSSAOParams.ssaoBlur;
		updateUniformBufferSSAOParams();
	}

	void toggleSSAOOnly()
	{
		uboSSAOParams.ssaoOnly = !uboSSAOParams.ssaoOnly;
		updateUniformBufferSSAOParams();
	}

	virtual void keyPressed(uint32_t keyCode)
	{
		switch (keyCode)
		{
		case KEY_F2:
		case GAMEPAD_BUTTON_A:
			toggleSSAO();
			break;
		case KEY_F3:
		case GAMEPAD_BUTTON_X:
			toggleSSAOBlur();
			break;
		case KEY_F4:
		case GAMEPAD_BUTTON_Y:
			toggleSSAOOnly();
			break;
		}
	}

	virtual void getOverlayText(VulkanTextOverlay *textOverlay)
	{
#if defined(__ANDROID__)
		textOverlay->addText("\"Button A\" to toggle SSAO", 5.0f, 85.0f, VulkanTextOverlay::alignLeft);
		textOverlay->addText("\"Button X\" to toggle SSAO blur", 5.0f, 100.0f, VulkanTextOverlay::alignLeft);
		textOverlay->addText("\"Button Y\" to toggle SSAO display", 5.0f, 115.0f, VulkanTextOverlay::alignLeft);
#else
		textOverlay->addText("\"F2\" to toggle SSAO", 5.0f, 85.0f, VulkanTextOverlay::alignLeft);
		textOverlay->addText("\"F3\" to toggle SSAO blur", 5.0f, 100.0f, VulkanTextOverlay::alignLeft);
		textOverlay->addText("\"F4\" to toggle SSAO display", 5.0f, 115.0f, VulkanTextOverlay::alignLeft);
#endif
	}
};

VULKAN_EXAMPLE_MAIN()
