/*
* Vulkan Example - Fullscreen radial blur (Single pass offscreen effect)
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

// Offscreen frame buffer properties
#define FB_DIM 512
#define FB_COLOR_FORMAT vk::Format::eR8G8B8A8Unorm

class VulkanExample : public VulkanExampleBase
{
public:
	bool blur = true;
	bool displayTexture = false;

	struct {
		vks::Texture2D gradient;
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
	} models;

	struct {
		vk::PipelineVertexInputStateCreateInfo inputState;
		std::vector<vk::VertexInputBindingDescription> bindingDescriptions;
		std::vector<vk::VertexInputAttributeDescription> attributeDescriptions;
	} vertices;

	struct {
		vks::Buffer scene;
		vks::Buffer blurParams;
	} uniformBuffers;

	struct UboVS {
		glm::mat4 projection;
		glm::mat4 model;
		float gradientPos = 0.0f;
	} uboScene;

	struct UboBlurParams {
		float radialBlurScale = 0.35f;
		float radialBlurStrength = 0.75f;
		glm::vec2 radialOrigin = glm::vec2(0.5f, 0.5f);
	} uboBlurParams;

	struct {
		vk::Pipeline radialBlur;
		vk::Pipeline colorPass;
		vk::Pipeline phongPass;
		vk::Pipeline offscreenDisplay;
	} pipelines;

	struct {
		vk::PipelineLayout radialBlur;
		vk::PipelineLayout scene;
	} pipelineLayouts;

	struct {
		vk::DescriptorSet scene;
		vk::DescriptorSet radialBlur;
	} descriptorSets;

	struct {
		vk::DescriptorSetLayout scene;
		vk::DescriptorSetLayout radialBlur;
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

	VulkanExample() : VulkanExampleBase(ENABLE_VALIDATION)
	{
		zoom = -10.0f;
		rotation = { -16.25f, -28.75f, 0.0f };
		timerSpeed *= 0.5f;
		enableTextOverlay = true;
		title = "Vulkan Example - Radial blur";
	}

	~VulkanExample()
	{
		// Clean up used Vulkan resources 
		// Note : Inherited destructor cleans up resources stored in base class

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

		device.destroyPipeline(pipelines.radialBlur);
		device.destroyPipeline(pipelines.phongPass);
		device.destroyPipeline(pipelines.colorPass);
		device.destroyPipeline(pipelines.offscreenDisplay);

		device.destroyPipelineLayout(pipelineLayouts.radialBlur);
		device.destroyPipelineLayout(pipelineLayouts.scene);

		device.destroyDescriptorSetLayout(descriptorSetLayouts.scene);
		device.destroyDescriptorSetLayout(descriptorSetLayouts.radialBlur);

		models.example.destroy();

		uniformBuffers.scene.destroy();
		uniformBuffers.blurParams.destroy();

		device.freeCommandBuffers(cmdPool, offscreenPass.commandBuffer);
		device.destroySemaphore(offscreenPass.semaphore);

		textures.gradient.destroy();
	}

	// Setup the offscreen framebuffer for rendering the blurred scene
	// The color attachment of this framebuffer will then be used to sample frame in the fragment shader of the final pass
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

		vk::Viewport viewport = vks::initializers::viewport((float)offscreenPass.width, (float)offscreenPass.height, 0.0f, 1.0f);
		offscreenPass.commandBuffer.setViewport(0, viewport);

		vk::Rect2D scissor = vks::initializers::rect2D(offscreenPass.width, offscreenPass.height, 0, 0);
		offscreenPass.commandBuffer.setScissor(0, scissor);

		offscreenPass.commandBuffer.beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);

		offscreenPass.commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayouts.scene, 0, descriptorSets.scene, nullptr);
		offscreenPass.commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.colorPass);

		std::vector<vk::DeviceSize> offsets = { 0 };
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

			vk::Rect2D scissor = vks::initializers::rect2D(width, height, 0, 0);
			drawCmdBuffers[i].setScissor(0, scissor);

			std::vector<vk::DeviceSize> offsets = { 0 };

			// 3D scene
			drawCmdBuffers[i].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayouts.scene, 0, descriptorSets.scene, nullptr);
			drawCmdBuffers[i].bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.phongPass);

			drawCmdBuffers[i].bindVertexBuffers(VERTEX_BUFFER_BIND_ID, models.example.vertices.buffer, offsets);
			drawCmdBuffers[i].bindIndexBuffer(models.example.indices.buffer, 0, vk::IndexType::eUint32);
			drawCmdBuffers[i].drawIndexed(models.example.indexCount, 1, 0, 0, 0);

			// Fullscreen triangle (clipped to a quad) with radial blur
			if (blur)
			{
				drawCmdBuffers[i].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayouts.radialBlur, 0, descriptorSets.radialBlur, nullptr);
				drawCmdBuffers[i].bindPipeline(vk::PipelineBindPoint::eGraphics, (displayTexture) ? pipelines.offscreenDisplay : pipelines.radialBlur);
				vkCmdDraw(drawCmdBuffers[i], 3, 1, 0, 0);
			}

			drawCmdBuffers[i].endRenderPass();

			drawCmdBuffers[i].end();
		}
	}

	void loadAssets()
	{
		models.example.loadFromFile(getAssetPath() + "models/glowsphere.dae", vertexLayout, 0.05f, vulkanDevice, queue);
		textures.gradient.loadFromFile(getAssetPath() + "textures/particle_gradient_rgba.ktx", vk::Format::eR8G8B8A8Unorm, vulkanDevice, queue);
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
		// Example uses three ubos and one image sampler
		std::vector<vk::DescriptorPoolSize> poolSizes =
		{
			vks::initializers::descriptorPoolSize(vk::DescriptorType::eUniformBuffer, 4),
			vks::initializers::descriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, 6)
		};

		vk::DescriptorPoolCreateInfo descriptorPoolInfo =
			vks::initializers::descriptorPoolCreateInfo(
				poolSizes.size(),
				poolSizes.data(),
				2);

		descriptorPool = device.createDescriptorPool(descriptorPoolInfo);
	}

	void setupDescriptorSetLayout()
	{
		std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings;
		vk::DescriptorSetLayoutCreateInfo descriptorLayout;
		vk::PipelineLayoutCreateInfo pPipelineLayoutCreateInfo;

		// Scene rendering
		setLayoutBindings =
		{
			// Binding 0: Vertex shader uniform buffer
			vks::initializers::descriptorSetLayoutBinding(
				vk::DescriptorType::eUniformBuffer,
				vk::ShaderStageFlagBits::eVertex,
				0),
			// Binding 1: Fragment shader image sampler
			vks::initializers::descriptorSetLayoutBinding(
				vk::DescriptorType::eCombinedImageSampler,
				vk::ShaderStageFlagBits::eFragment,
				1),
			// Binding 2: Fragment shader uniform buffer
			vks::initializers::descriptorSetLayoutBinding(
				vk::DescriptorType::eUniformBuffer,
				vk::ShaderStageFlagBits::eFragment,
				2)
		};
		descriptorLayout = vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
		descriptorSetLayouts.scene = device.createDescriptorSetLayout(descriptorLayout);
		pPipelineLayoutCreateInfo = vks::initializers::pipelineLayoutCreateInfo(&descriptorSetLayouts.scene, 1);
		pipelineLayouts.scene = device.createPipelineLayout(pPipelineLayoutCreateInfo);

		// Fullscreen radial blur
		setLayoutBindings =
		{
			// Binding 0 : Vertex shader uniform buffer
			vks::initializers::descriptorSetLayoutBinding(
				vk::DescriptorType::eUniformBuffer,
				vk::ShaderStageFlagBits::eFragment,
				0),
			// Binding 0: Fragment shader image sampler
			vks::initializers::descriptorSetLayoutBinding(
				vk::DescriptorType::eCombinedImageSampler,
				vk::ShaderStageFlagBits::eFragment,
				1)
		};
		descriptorLayout = vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
		descriptorSetLayouts.radialBlur = device.createDescriptorSetLayout(descriptorLayout);
		pPipelineLayoutCreateInfo = vks::initializers::pipelineLayoutCreateInfo(&descriptorSetLayouts.radialBlur, 1);
		pipelineLayouts.radialBlur = device.createPipelineLayout(pPipelineLayoutCreateInfo);
	}

	void setupDescriptorSet()
	{
		vk::DescriptorSetAllocateInfo descriptorSetAllocInfo;

		// Scene rendering
		descriptorSetAllocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorPool, &descriptorSetLayouts.scene, 1);
		descriptorSets.scene = device.allocateDescriptorSets(descriptorSetAllocInfo)[0];

		std::vector<vk::WriteDescriptorSet> offScreenWriteDescriptorSets =
		{
			// Binding 0: Vertex shader uniform buffer
			vks::initializers::writeDescriptorSet(
				descriptorSets.scene,
				vk::DescriptorType::eUniformBuffer,
				0,
				&uniformBuffers.scene.descriptor),
			// Binding 1: Color gradient sampler
			vks::initializers::writeDescriptorSet(
				descriptorSets.scene, 
				vk::DescriptorType::eCombinedImageSampler, 
				1, 
				&textures.gradient.descriptor),
		};
		device.updateDescriptorSets(offScreenWriteDescriptorSets, nullptr);

		// Fullscreen radial blur
		descriptorSetAllocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorPool, &descriptorSetLayouts.radialBlur, 1);
		descriptorSets.radialBlur = device.allocateDescriptorSets(descriptorSetAllocInfo)[0];

		std::vector<vk::WriteDescriptorSet> writeDescriptorSets =
		{
			// Binding 0: Vertex shader uniform buffer
			vks::initializers::writeDescriptorSet(
				descriptorSets.radialBlur,
				vk::DescriptorType::eUniformBuffer,
				0,
				&uniformBuffers.blurParams.descriptor),
			// Binding 0: Fragment shader texture sampler
			vks::initializers::writeDescriptorSet(
				descriptorSets.radialBlur, 
				vk::DescriptorType::eCombinedImageSampler,
				1, 
				&offscreenPass.descriptor),
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
				vk::FrontFace::eCounterClockwise);

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

		std::array<vk::PipelineShaderStageCreateInfo, 2> shaderStages;

		vk::GraphicsPipelineCreateInfo pipelineCreateInfo =
			vks::initializers::pipelineCreateInfo(
				pipelineLayouts.radialBlur,
				renderPass);

		pipelineCreateInfo.pInputAssemblyState = &inputAssemblyState;
		pipelineCreateInfo.pRasterizationState = &rasterizationState;
		pipelineCreateInfo.pColorBlendState = &colorBlendState;
		pipelineCreateInfo.pMultisampleState = &multisampleState;
		pipelineCreateInfo.pViewportState = &viewportState;
		pipelineCreateInfo.pDepthStencilState = &depthStencilState;
		pipelineCreateInfo.pDynamicState = &dynamicState;
		pipelineCreateInfo.stageCount = shaderStages.size();
		pipelineCreateInfo.pStages = shaderStages.data();

		// Radial blur pipeline
		shaderStages[0] = loadShader(getAssetPath() + "shaders/radialblur/radialblur.vert.spv", vk::ShaderStageFlagBits::eVertex);
		shaderStages[1] = loadShader(getAssetPath() + "shaders/radialblur/radialblur.frag.spv", vk::ShaderStageFlagBits::eFragment);
		// Empty vertex input state
		vk::PipelineVertexInputStateCreateInfo emptyInputState = vks::initializers::pipelineVertexInputStateCreateInfo();
		pipelineCreateInfo.pVertexInputState = &emptyInputState;
		pipelineCreateInfo.layout = pipelineLayouts.radialBlur;
		// Additive blending
		blendAttachmentState.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
		blendAttachmentState.blendEnable = VK_TRUE;
		blendAttachmentState.colorBlendOp = vk::BlendOp::eAdd;
		blendAttachmentState.srcColorBlendFactor = vk::BlendFactor::eOne;
		blendAttachmentState.dstColorBlendFactor = vk::BlendFactor::eOne;
		blendAttachmentState.alphaBlendOp = vk::BlendOp::eAdd;
		blendAttachmentState.srcAlphaBlendFactor = vk::BlendFactor::eSrcAlpha;
		blendAttachmentState.dstAlphaBlendFactor = vk::BlendFactor::eDstAlpha;
		pipelines.radialBlur = device.createGraphicsPipelines(pipelineCache, pipelineCreateInfo)[0];

		// No blending (for debug display)
		blendAttachmentState.blendEnable = VK_FALSE;
		pipelines.offscreenDisplay = device.createGraphicsPipelines(pipelineCache, pipelineCreateInfo)[0];

		// Phong pass
		pipelineCreateInfo.layout = pipelineLayouts.scene;
		shaderStages[0] = loadShader(getAssetPath() + "shaders/radialblur/phongpass.vert.spv", vk::ShaderStageFlagBits::eVertex);
		shaderStages[1] = loadShader(getAssetPath() + "shaders/radialblur/phongpass.frag.spv", vk::ShaderStageFlagBits::eFragment);
		pipelineCreateInfo.pVertexInputState = &vertices.inputState;
		blendAttachmentState.blendEnable = VK_FALSE;
		depthStencilState.depthWriteEnable = VK_TRUE;
		pipelines.phongPass = device.createGraphicsPipelines(pipelineCache, pipelineCreateInfo)[0];

		// Color only pass (offscreen blur base)
		shaderStages[0] = loadShader(getAssetPath() + "shaders/radialblur/colorpass.vert.spv", vk::ShaderStageFlagBits::eVertex);
		shaderStages[1] = loadShader(getAssetPath() + "shaders/radialblur/colorpass.frag.spv", vk::ShaderStageFlagBits::eFragment);
		pipelineCreateInfo.renderPass = offscreenPass.renderPass;
		pipelines.colorPass = device.createGraphicsPipelines(pipelineCache, pipelineCreateInfo)[0];
	}

	// Prepare and initialize uniform buffer containing shader uniforms
	void prepareUniformBuffers()
	{
		// Phong and color pass vertex shader uniform buffer
		vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eUniformBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			&uniformBuffers.scene,
			sizeof(uboScene));

		// Fullscreen radial blur parameters
		vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eUniformBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			&uniformBuffers.blurParams,
			sizeof(uboBlurParams),
			&uboBlurParams);

		// Map persistent
		uniformBuffers.scene.map();
		uniformBuffers.blurParams.map();

		updateUniformBuffersScene();
	}

	// Update uniform buffers for rendering the 3D scene
	void updateUniformBuffersScene()
	{
		uboScene.projection = glm::perspective(glm::radians(45.0f), (float)width / (float)height, 1.0f, 256.0f);
		glm::mat4 viewMatrix = glm::translate(glm::mat4(), glm::vec3(0.0f, 0.0f, zoom));

		uboScene.model = glm::mat4();
		uboScene.model = viewMatrix * glm::translate(uboScene.model, cameraPos);
		uboScene.model = glm::rotate(uboScene.model, glm::radians(rotation.x), glm::vec3(1.0f, 0.0f, 0.0f));
		uboScene.model = glm::rotate(uboScene.model, glm::radians(rotation.y), glm::vec3(0.0f, 1.0f, 0.0f));
		uboScene.model = glm::rotate(uboScene.model, glm::radians(timer * 360.0f), glm::vec3(0.0f, 1.0f, 0.0f));
		uboScene.model = glm::rotate(uboScene.model, glm::radians(rotation.z), glm::vec3(0.0f, 0.0f, 1.0f));

		if (!paused)
		{
			uboScene.gradientPos += frameTimer * 0.1f;
		}

		memcpy(uniformBuffers.scene.mapped, &uboScene, sizeof(uboScene));
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
			updateUniformBuffersScene();
		}
	}

	virtual void viewChanged()
	{
		updateUniformBuffersScene();
	}

	void toggleBlur()
	{
		blur = !blur;
		updateUniformBuffersScene();
		reBuildCommandBuffers();
	}

	void toggleTextureDisplay()
	{
		displayTexture = !displayTexture;
		reBuildCommandBuffers();
	}

	virtual void keyPressed(uint32_t keyCode)
	{
		switch (keyCode)
		{
		case KEY_B:
		case GAMEPAD_BUTTON_A:
			toggleBlur();
			break;
		case KEY_T:
		case GAMEPAD_BUTTON_X:
			toggleTextureDisplay();
			break;
		}
	}

	virtual void getOverlayText(VulkanTextOverlay *textOverlay)
	{
#if defined(__ANDROID__)
		textOverlay->addText("Press \"Button A\" to toggle blur", 5.0f, 85.0f, VulkanTextOverlay::alignLeft);
		textOverlay->addText("Press \"Button X\" to display offscreen texture", 5.0f, 105.0f, VulkanTextOverlay::alignLeft);
#else
		textOverlay->addText("Press \"B\" to toggle blur", 5.0f, 85.0f, VulkanTextOverlay::alignLeft);
		textOverlay->addText("Press \"T\" to display offscreen texture", 5.0f, 105.0f, VulkanTextOverlay::alignLeft);
#endif
	}
};

VULKAN_EXAMPLE_MAIN()
