/*
* Vulkan Example - Using subpasses for G-Buffer compositing
*
* Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include "vulkanexamplebase.h"
#include "VulkanTexture.hpp"
#include "VulkanModel.hpp"

#include <random>

#define VERTEX_BUFFER_BIND_ID 0
#define ENABLE_VALIDATION false

#define NUM_LIGHTS 64

class VulkanExample : public VulkanExampleBase
{
public:
	struct {
		vks::Texture2D glass;
	} textures;

	// Vertex layout for the models
	vks::VertexLayout vertexLayout = vks::VertexLayout({
		vks::VERTEX_COMPONENT_POSITION,
		vks::VERTEX_COMPONENT_COLOR,
		vks::VERTEX_COMPONENT_NORMAL,
		vks::VERTEX_COMPONENT_UV,
	});

	struct {
		vks::Model scene;
		vks::Model transparent;
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
	} uboGBuffer;

	struct Light {
		glm::vec4 position;
		glm::vec3 color;
		float radius;
	};

	struct {
		glm::vec4 viewPos;
		Light lights[NUM_LIGHTS];
	} uboLights;

	struct {
		vks::Buffer GBuffer;
		vks::Buffer lights;
	} uniformBuffers;

	struct {
		vk::Pipeline offscreen;
		vk::Pipeline composition;
		vk::Pipeline transparent;
	} pipelines;

	struct {
		vk::PipelineLayout offscreen;
		vk::PipelineLayout composition;
		vk::PipelineLayout transparent;
	} pipelineLayouts;

	struct {
		vk::DescriptorSet scene;
		vk::DescriptorSet composition;
		vk::DescriptorSet transparent;
	} descriptorSets;

	struct {
		vk::DescriptorSetLayout scene;
		vk::DescriptorSetLayout composition;
		vk::DescriptorSetLayout transparent;
	} descriptorSetLayouts;

	// G-Buffer framebuffer attachments
	struct FrameBufferAttachment {
		vk::Image image;
		vk::DeviceMemory mem;
		vk::ImageView view;
		vk::Format format;
	};
	struct Attachments {
		FrameBufferAttachment position, normal, albedo;
	} attachments;
	
	VulkanExample() : VulkanExampleBase(ENABLE_VALIDATION)
	{
		enableTextOverlay = false;
		title = "Vulkan Example - Subpasses";
		camera.type = Camera::CameraType::firstperson;
		camera.movementSpeed = 5.0f;
#ifndef __ANDROID__
		camera.rotationSpeed = 0.25f;
#endif  
		camera.setPosition(glm::vec3(-3.2f, 1.0f, 5.9f));
		camera.setRotation(glm::vec3(0.5f, 210.05f, 0.0f));
		camera.setPerspective(60.0f, (float)width / (float)height, 0.1f, 256.0f);
	}

	~VulkanExample()
	{
		// Clean up used Vulkan resources 
		// Note : Inherited destructor cleans up resources stored in base class

		device.destroyImageView(attachments.position.view);
		device.destroyImage(attachments.position.image);
		device.freeMemory(attachments.position.mem);

		device.destroyImageView(attachments.normal.view);
		device.destroyImage(attachments.normal.image);
		device.freeMemory(attachments.normal.mem);

		device.destroyImageView(attachments.albedo.view);
		device.destroyImage(attachments.albedo.image);
		device.freeMemory(attachments.albedo.mem);

		device.destroyPipeline(pipelines.offscreen);
		device.destroyPipeline(pipelines.composition);
		device.destroyPipeline(pipelines.transparent);

		device.destroyPipelineLayout(pipelineLayouts.offscreen);
		device.destroyPipelineLayout(pipelineLayouts.composition);
		device.destroyPipelineLayout(pipelineLayouts.transparent);

		device.destroyDescriptorSetLayout(descriptorSetLayouts.scene);
		device.destroyDescriptorSetLayout(descriptorSetLayouts.composition);
		device.destroyDescriptorSetLayout(descriptorSetLayouts.transparent);

		textures.glass.destroy();
		models.scene.destroy();
		models.transparent.destroy();
		uniformBuffers.GBuffer.destroy();
		uniformBuffers.lights.destroy();
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
	void createAttachment(vk::Format format, vk::ImageUsageFlags usage, FrameBufferAttachment *attachment)
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

		//assert(aspectMask > 0);

		vk::ImageCreateInfo image = vks::initializers::imageCreateInfo();
		image.imageType = vk::ImageType::e2D;
		image.format = format;
		image.extent.width = width;
		image.extent.height = height;
		image.extent.depth = 1;
		image.mipLevels = 1;
		image.arrayLayers = 1;
		image.samples = vk::SampleCountFlagBits::e1;
		image.tiling = vk::ImageTiling::eOptimal;
		image.usage = usage | vk::ImageUsageFlagBits::eInputAttachment;	// vk::ImageUsageFlagBits::eInputAttachment flag is required for input attachments;
		image.initialLayout = vk::ImageLayout::eUndefined;

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

	// Create color attachments for the G-Buffer components
	void createGBufferAttachments()
	{
		createAttachment(vk::Format::eR16G16B16A16Sfloat, vk::ImageUsageFlagBits::eColorAttachment, &attachments.position);	// (World space) Positions		
		createAttachment(vk::Format::eR16G16B16A16Sfloat, vk::ImageUsageFlagBits::eColorAttachment, &attachments.normal);		// (World space) Normals		
		createAttachment(vk::Format::eR8G8B8A8Unorm, vk::ImageUsageFlagBits::eColorAttachment, &attachments.albedo);			// Albedo (color)
	}

	// Override framebuffer setup from base class
	// Deferred components will be used as frame buffer attachments
	void setupFrameBuffer()
	{
		vk::ImageView attachments[5];

		vk::FramebufferCreateInfo frameBufferCreateInfo = {};
		frameBufferCreateInfo.renderPass = renderPass;
		frameBufferCreateInfo.attachmentCount = 5;
		frameBufferCreateInfo.pAttachments = attachments;
		frameBufferCreateInfo.width = width;
		frameBufferCreateInfo.height = height;
		frameBufferCreateInfo.layers = 1;

		// Create frame buffers for every swap chain image
		frameBuffers.resize(swapChain.imageCount);
		for (uint32_t i = 0; i < frameBuffers.size(); i++)
		{
			attachments[0] = swapChain.buffers[i].view;
			attachments[1] = this->attachments.position.view;
			attachments[2] = this->attachments.normal.view;
			attachments[3] = this->attachments.albedo.view;
			attachments[4] = depthStencil.view;
			frameBuffers[i] = device.createFramebuffer(frameBufferCreateInfo);
		}
	}

	// Override render pass setup from base class
	void setupRenderPass()
	{
		createGBufferAttachments(); 

		std::array<vk::AttachmentDescription, 5> attachments{};
		// Color attachment
		attachments[0].format = swapChain.colorFormat;
		attachments[0].samples = vk::SampleCountFlagBits::e1;
		attachments[0].loadOp = vk::AttachmentLoadOp::eClear;
		attachments[0].storeOp = vk::AttachmentStoreOp::eStore;
		attachments[0].stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
		attachments[0].stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
		attachments[0].initialLayout = vk::ImageLayout::eUndefined;
		attachments[0].finalLayout = vk::ImageLayout::ePresentSrcKHR;

		// Deferred attachments
		// Position
		attachments[1].format = this->attachments.position.format;
		attachments[1].samples = vk::SampleCountFlagBits::e1;
		attachments[1].loadOp = vk::AttachmentLoadOp::eClear;
		attachments[1].storeOp = vk::AttachmentStoreOp::eDontCare;
		attachments[1].stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
		attachments[1].stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
		attachments[1].initialLayout = vk::ImageLayout::eUndefined;
		attachments[1].finalLayout = vk::ImageLayout::eColorAttachmentOptimal;
		// Normals
		attachments[2].format = this->attachments.normal.format;
		attachments[2].samples = vk::SampleCountFlagBits::e1;
		attachments[2].loadOp = vk::AttachmentLoadOp::eClear;
		attachments[2].storeOp = vk::AttachmentStoreOp::eDontCare;
		attachments[2].stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
		attachments[2].stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
		attachments[2].initialLayout = vk::ImageLayout::eUndefined;
		attachments[2].finalLayout = vk::ImageLayout::eColorAttachmentOptimal;
		// Albedo
		attachments[3].format = this->attachments.albedo.format;
		attachments[3].samples = vk::SampleCountFlagBits::e1;
		attachments[3].loadOp = vk::AttachmentLoadOp::eClear;
		attachments[3].storeOp = vk::AttachmentStoreOp::eDontCare;
		attachments[3].stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
		attachments[3].stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
		attachments[3].initialLayout = vk::ImageLayout::eUndefined;
		attachments[3].finalLayout = vk::ImageLayout::eColorAttachmentOptimal;
		// Depth attachment
		attachments[4].format = depthFormat;
		attachments[4].samples = vk::SampleCountFlagBits::e1;
		attachments[4].loadOp = vk::AttachmentLoadOp::eClear;
		attachments[4].storeOp = vk::AttachmentStoreOp::eDontCare;
		attachments[4].stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
		attachments[4].stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
		attachments[4].initialLayout = vk::ImageLayout::eUndefined;
		attachments[4].finalLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal;

		// Three subpasses
		std::array<vk::SubpassDescription,3> subpassDescriptions{};

		// First subpass: Fill G-Buffer components
		// ----------------------------------------------------------------------------------------

		vk::AttachmentReference colorReferences[4];
		colorReferences[0] = { 0, vk::ImageLayout::eColorAttachmentOptimal };
		colorReferences[1] = { 1, vk::ImageLayout::eColorAttachmentOptimal };
		colorReferences[2] = { 2, vk::ImageLayout::eColorAttachmentOptimal };
		colorReferences[3] = { 3, vk::ImageLayout::eColorAttachmentOptimal };
		vk::AttachmentReference depthReference = { 4, vk::ImageLayout::eDepthStencilAttachmentOptimal };

		subpassDescriptions[0].pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
		subpassDescriptions[0].colorAttachmentCount = 4;
		subpassDescriptions[0].pColorAttachments = colorReferences;
		subpassDescriptions[0].pDepthStencilAttachment = &depthReference;

		// Second subpass: Final composition (using G-Buffer components)
		// ----------------------------------------------------------------------------------------

		vk::AttachmentReference colorReference = { 0, vk::ImageLayout::eColorAttachmentOptimal };

		vk::AttachmentReference inputReferences[3];
		inputReferences[0] = { 1, vk::ImageLayout::eShaderReadOnlyOptimal };
		inputReferences[1] = { 2, vk::ImageLayout::eShaderReadOnlyOptimal };
		inputReferences[2] = { 3, vk::ImageLayout::eShaderReadOnlyOptimal };

		//uint32_t preserveAttachmentIndex = 1;

		subpassDescriptions[1].pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
		subpassDescriptions[1].colorAttachmentCount = 1;
		subpassDescriptions[1].pColorAttachments = &colorReference;
		subpassDescriptions[1].pDepthStencilAttachment = &depthReference;
		// Use the color attachments filled in the first pass as input attachments
		subpassDescriptions[1].inputAttachmentCount = 3;
		subpassDescriptions[1].pInputAttachments = inputReferences;

		// Third subpass: Forward transparency
		// ----------------------------------------------------------------------------------------
		colorReference = { 0, vk::ImageLayout::eColorAttachmentOptimal };

		inputReferences[0] = { 1, vk::ImageLayout::eShaderReadOnlyOptimal };

		subpassDescriptions[2].pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
		subpassDescriptions[2].colorAttachmentCount = 1;
		subpassDescriptions[2].pColorAttachments = &colorReference;
		subpassDescriptions[2].pDepthStencilAttachment = &depthReference;
		// Use the color/depth attachments filled in the first pass as input attachments
		subpassDescriptions[2].inputAttachmentCount = 1;
		subpassDescriptions[2].pInputAttachments = inputReferences;

		// Subpass dependencies for layout transitions
		std::array<vk::SubpassDependency, 4> dependencies;

		dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
		dependencies[0].dstSubpass = 0;
		dependencies[0].srcStageMask = vk::PipelineStageFlagBits::eBottomOfPipe;
		dependencies[0].dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
		dependencies[0].srcAccessMask = vk::AccessFlagBits::eMemoryRead;
		dependencies[0].dstAccessMask = vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite;
		dependencies[0].dependencyFlags = vk::DependencyFlagBits::eByRegion;

		// This dependency transitions the input attachment from color attachment to shader read
		dependencies[1].srcSubpass = 0;
		dependencies[1].dstSubpass = 1;
		dependencies[1].srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
		dependencies[1].dstStageMask = vk::PipelineStageFlagBits::eFragmentShader;
		dependencies[1].srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite;
		dependencies[1].dstAccessMask = vk::AccessFlagBits::eShaderRead;
		dependencies[1].dependencyFlags = vk::DependencyFlagBits::eByRegion;

		dependencies[2].srcSubpass = 1;
		dependencies[2].dstSubpass = 2;
		dependencies[2].srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
		dependencies[2].dstStageMask = vk::PipelineStageFlagBits::eFragmentShader;
		dependencies[2].srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite;
		dependencies[2].dstAccessMask = vk::AccessFlagBits::eShaderRead;
		dependencies[2].dependencyFlags = vk::DependencyFlagBits::eByRegion;

		dependencies[3].srcSubpass = 0;
		dependencies[3].dstSubpass = VK_SUBPASS_EXTERNAL;
		dependencies[3].srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
		dependencies[3].dstStageMask = vk::PipelineStageFlagBits::eBottomOfPipe;
		dependencies[3].srcAccessMask = vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite;
		dependencies[3].dstAccessMask = vk::AccessFlagBits::eMemoryRead;
		dependencies[3].dependencyFlags = vk::DependencyFlagBits::eByRegion;

		vk::RenderPassCreateInfo renderPassInfo = {};

		renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
		renderPassInfo.pAttachments = attachments.data();
		renderPassInfo.subpassCount = static_cast<uint32_t>(subpassDescriptions.size());
		renderPassInfo.pSubpasses = subpassDescriptions.data();
		renderPassInfo.dependencyCount = static_cast<uint32_t>(dependencies.size());
		renderPassInfo.pDependencies = dependencies.data();

		renderPass = device.createRenderPass(renderPassInfo);
	}

	void buildCommandBuffers()
	{
		vk::CommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();

		vk::ClearValue clearValues[5];
		clearValues[0].color = vk::ClearColorValue{ std::array<float, 4>{ 0.0f, 0.0f, 0.0f, 0.0f } };
		clearValues[1].color = vk::ClearColorValue{ std::array<float, 4>{ 0.0f, 0.0f, 0.0f, 0.0f } };
		clearValues[2].color = vk::ClearColorValue{ std::array<float, 4>{ 0.0f, 0.0f, 0.0f, 0.0f } };
		clearValues[3].color = vk::ClearColorValue{ std::array<float, 4>{ 0.0f, 0.0f, 0.0f, 0.0f } };
		clearValues[4].depthStencil = vk::ClearDepthStencilValue{ 1.0f, 0 };

		vk::RenderPassBeginInfo renderPassBeginInfo = vks::initializers::renderPassBeginInfo();
		renderPassBeginInfo.renderPass = renderPass;
		renderPassBeginInfo.renderArea.offset.x = 0;
		renderPassBeginInfo.renderArea.offset.y = 0;
		renderPassBeginInfo.renderArea.extent.width = width;
		renderPassBeginInfo.renderArea.extent.height = height;
		renderPassBeginInfo.clearValueCount = 5;
		renderPassBeginInfo.pClearValues = clearValues;

		for (uint32_t i = 0; i < drawCmdBuffers.size(); ++i)
		{
			// Set target frame buffer
			renderPassBeginInfo.framebuffer = frameBuffers[i];

			drawCmdBuffers[i].begin(cmdBufInfo);

			// First sub pass
			// Renders the components of the scene to the G-Buffer atttachments

			drawCmdBuffers[i].beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);

			vk::Viewport viewport = vks::initializers::viewport((float)width, (float)height, 0.0f, 1.0f);
			drawCmdBuffers[i].setViewport(0, viewport);

			vk::Rect2D scissor = vks::initializers::rect2D(width, height, 0, 0);
			drawCmdBuffers[i].setScissor(0, scissor);

			std::vector<vk::DeviceSize> offsets = { 0 };

			drawCmdBuffers[i].bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.offscreen);

			drawCmdBuffers[i].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayouts.offscreen, 0, descriptorSets.scene, nullptr);
			drawCmdBuffers[i].bindVertexBuffers(VERTEX_BUFFER_BIND_ID, models.scene.vertices.buffer, offsets);
			drawCmdBuffers[i].bindIndexBuffer(models.scene.indices.buffer, 0, vk::IndexType::eUint32);
			drawCmdBuffers[i].drawIndexed(models.scene.indexCount, 1, 0, 0, 0);

			// Second sub pass
			// This subpass will use the G-Buffer components that have been filled in the first subpass as input attachment for the final compositing

			drawCmdBuffers[i].nextSubpass(vk::SubpassContents::eInline);

			drawCmdBuffers[i].bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.composition);
			drawCmdBuffers[i].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayouts.composition, 0, descriptorSets.composition, nullptr);
			vkCmdDraw(drawCmdBuffers[i], 3, 1, 0, 0);

			// Third subpass
			// Render transparent geometry using a forward pass that compares against depth generted during G-Buffer fill
			drawCmdBuffers[i].nextSubpass(vk::SubpassContents::eInline);

			drawCmdBuffers[i].bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.transparent);
			drawCmdBuffers[i].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayouts.transparent, 0, descriptorSets.transparent, nullptr);
			drawCmdBuffers[i].bindVertexBuffers(VERTEX_BUFFER_BIND_ID, models.transparent.vertices.buffer, offsets);
			drawCmdBuffers[i].bindIndexBuffer(models.transparent.indices.buffer, 0, vk::IndexType::eUint32);
			drawCmdBuffers[i].drawIndexed(models.transparent.indexCount, 1, 0, 0, 0);

			drawCmdBuffers[i].endRenderPass();

			drawCmdBuffers[i].end();
		}
	}

	void loadAssets()
	{
		models.scene.loadFromFile(getAssetPath() + "models/samplebuilding.dae", vertexLayout, 1.0f, vulkanDevice, queue);
		models.transparent.loadFromFile(getAssetPath() + "models/samplebuilding_glass.dae", vertexLayout, 1.0f, vulkanDevice, queue);
		// Textures
		if (vulkanDevice->features.textureCompressionBC) {
			textures.glass.loadFromFile(getAssetPath() + "textures/colored_glass_bc3_unorm.ktx", vk::Format::eBc3UnormBlock, vulkanDevice, queue);
		}
		else if (vulkanDevice->features.textureCompressionASTC_LDR) {
			textures.glass.loadFromFile(getAssetPath() + "textures/colored_glass_astc_8x8_unorm.ktx", vk::Format::eAstc8x8UnormBlock, vulkanDevice, queue);
		}
		else if (vulkanDevice->features.textureCompressionETC2) {
			textures.glass.loadFromFile(getAssetPath() + "textures/colored_glass_etc2_unorm.ktx", vk::Format::eEtc2R8G8B8A8UnormBlock, vulkanDevice, queue);
		}
		else {
			vks::tools::exitFatal("Device does not support any compressed texture format!", "Error");
		}
	}

	void setupVertexDescriptions()
	{
		// Binding description
		vertices.bindingDescriptions = {
			vks::initializers::vertexInputBindingDescription(
				VERTEX_BUFFER_BIND_ID,
				vertexLayout.stride(),
				vk::VertexInputRate::eVertex),
		};

		// Attribute descriptions
		vertices.attributeDescriptions = {
			// Location 0: Position
			vks::initializers::vertexInputAttributeDescription(
				VERTEX_BUFFER_BIND_ID,
				0,
				vk::Format::eR32G32B32Sfloat,
				0),
			// Location 1: Color
			vks::initializers::vertexInputAttributeDescription(
				VERTEX_BUFFER_BIND_ID,
				1,
				vk::Format::eR32G32B32Sfloat,
				sizeof(float) * 3),
			// Location 2: Normal
			vks::initializers::vertexInputAttributeDescription(
				VERTEX_BUFFER_BIND_ID,
				2,
				vk::Format::eR32G32B32Sfloat,
				sizeof(float) * 6),
			// Location 3: UV
			vks::initializers::vertexInputAttributeDescription(
				VERTEX_BUFFER_BIND_ID,
				3,
				vk::Format::eR32G32Sfloat,
				sizeof(float) * 9),
		};

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
			vks::initializers::descriptorPoolSize(vk::DescriptorType::eUniformBuffer, 9),
			vks::initializers::descriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, 9),
			vks::initializers::descriptorPoolSize(vk::DescriptorType::eInputAttachment, 4),
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
		// Deferred shading layout
		std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings =
		{
			// Binding 0 : Vertex shader uniform buffer
			vks::initializers::descriptorSetLayoutBinding(
				vk::DescriptorType::eUniformBuffer,
				vk::ShaderStageFlagBits::eVertex,
				0)
		};

		vk::DescriptorSetLayoutCreateInfo descriptorLayout =
			vks::initializers::descriptorSetLayoutCreateInfo(
				setLayoutBindings);

		descriptorSetLayouts.scene = device.createDescriptorSetLayout(descriptorLayout);

		vk::PipelineLayoutCreateInfo pPipelineLayoutCreateInfo =
			vks::initializers::pipelineLayoutCreateInfo(
				&descriptorSetLayouts.scene,
				1);

		// Offscreen (scene) rendering pipeline layout
		pipelineLayouts.offscreen = device.createPipelineLayout(pPipelineLayoutCreateInfo);
	}

	void setupDescriptorSet()
	{
		std::vector<vk::WriteDescriptorSet> writeDescriptorSets;

		vk::DescriptorSetAllocateInfo allocInfo =
			vks::initializers::descriptorSetAllocateInfo(
				descriptorPool,
				&descriptorSetLayouts.scene,
				1);

		descriptorSets.scene = device.allocateDescriptorSets(allocInfo)[0];
		writeDescriptorSets =
		{
			// Binding 0: Vertex shader uniform buffer
			vks::initializers::writeDescriptorSet(
				descriptorSets.scene,
				vk::DescriptorType::eUniformBuffer,
				0,
				&uniformBuffers.GBuffer.descriptor)
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
				pipelineLayouts.offscreen,
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
		pipelineCreateInfo.subpass = 0;

		vk::PipelineColorBlendAttachmentState def = vks::initializers::pipelineColorBlendAttachmentState(
			vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA, 
			VK_FALSE);
		std::array<vk::PipelineColorBlendAttachmentState, 4> blendAttachmentStates = { def, def, def, def };

		colorBlendState.attachmentCount = static_cast<uint32_t>(blendAttachmentStates.size());
		colorBlendState.pAttachments = blendAttachmentStates.data();

		// Offscreen scene rendering pipeline
		shaderStages[0] = loadShader(getAssetPath() + "shaders/subpasses/gbuffer.vert.spv", vk::ShaderStageFlagBits::eVertex);
		shaderStages[1] = loadShader(getAssetPath() + "shaders/subpasses/gbuffer.frag.spv", vk::ShaderStageFlagBits::eFragment);
	
		pipelines.offscreen = device.createGraphicsPipelines(pipelineCache, pipelineCreateInfo)[0];
	}

	// Create the Vulkan objects used in the composition pass (descriptor sets, pipelines, etc.)
	void prepareCompositionPass()
	{
		// Descriptor set layout
		std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings =
		{
			// Binding 0: Position input attachment 
			vks::initializers::descriptorSetLayoutBinding(
				vk::DescriptorType::eInputAttachment,
				vk::ShaderStageFlagBits::eFragment,
				0),
			// Binding 1: Normal input attachment 
			vks::initializers::descriptorSetLayoutBinding(
				vk::DescriptorType::eInputAttachment,
				vk::ShaderStageFlagBits::eFragment,
				1),
			// Binding 2: Albedo input attachment 
			vks::initializers::descriptorSetLayoutBinding(
				vk::DescriptorType::eInputAttachment,
				vk::ShaderStageFlagBits::eFragment,
				2),
			// Binding 3: Light positions
			vks::initializers::descriptorSetLayoutBinding(
				vk::DescriptorType::eUniformBuffer,
				vk::ShaderStageFlagBits::eFragment,
				3),
		};

		vk::DescriptorSetLayoutCreateInfo descriptorLayout =
			vks::initializers::descriptorSetLayoutCreateInfo(
				setLayoutBindings);

		descriptorSetLayouts.composition = device.createDescriptorSetLayout(descriptorLayout);

		// Pipeline layout
		vk::PipelineLayoutCreateInfo pPipelineLayoutCreateInfo = 
			vks::initializers::pipelineLayoutCreateInfo(&descriptorSetLayouts.composition, 1);

		pipelineLayouts.composition = device.createPipelineLayout(pPipelineLayoutCreateInfo);

		// Descriptor sets
		vk::DescriptorSetAllocateInfo allocInfo =
			vks::initializers::descriptorSetAllocateInfo(descriptorPool, &descriptorSetLayouts.composition, 1);

		descriptorSets.composition = device.allocateDescriptorSets(allocInfo)[0];

		// Image descriptors for the offscreen color attachments
		vk::DescriptorImageInfo texDescriptorPosition =
			vks::initializers::descriptorImageInfo(
				nullptr,
				attachments.position.view,
				vk::ImageLayout::eShaderReadOnlyOptimal);

		vk::DescriptorImageInfo texDescriptorNormal =
			vks::initializers::descriptorImageInfo(
				nullptr,
				attachments.normal.view,
				vk::ImageLayout::eShaderReadOnlyOptimal);

		vk::DescriptorImageInfo texDescriptorAlbedo =
			vks::initializers::descriptorImageInfo(
				nullptr,
				attachments.albedo.view,
				vk::ImageLayout::eShaderReadOnlyOptimal);

		std::vector<vk::WriteDescriptorSet> writeDescriptorSets = {
			// Binding 0: Position texture target
			vks::initializers::writeDescriptorSet(
				descriptorSets.composition,
				vk::DescriptorType::eInputAttachment,
				0,
				&texDescriptorPosition),
			// Binding 1: Normals texture target
			vks::initializers::writeDescriptorSet(
				descriptorSets.composition,
				vk::DescriptorType::eInputAttachment,
				1,
				&texDescriptorNormal),
			// Binding 2: Albedo texture target
			vks::initializers::writeDescriptorSet(
				descriptorSets.composition,
				vk::DescriptorType::eInputAttachment,
				2,
				&texDescriptorAlbedo),
			// Binding 4: Fragment shader lights
			vks::initializers::writeDescriptorSet(
				descriptorSets.composition,
				vk::DescriptorType::eUniformBuffer,
				3,
				&uniformBuffers.lights.descriptor),
		};

		device.updateDescriptorSets(writeDescriptorSets, nullptr);

		// Pipeline
		vk::PipelineInputAssemblyStateCreateInfo inputAssemblyState =
			vks::initializers::pipelineInputAssemblyStateCreateInfo(vk::PrimitiveTopology::eTriangleList, vk::PipelineInputAssemblyStateCreateFlags(), VK_FALSE);

		vk::PipelineRasterizationStateCreateInfo rasterizationState =
			vks::initializers::pipelineRasterizationStateCreateInfo(vk::PolygonMode::eFill, vk::CullModeFlagBits::eNone, vk::FrontFace::eClockwise);

		vk::PipelineColorBlendAttachmentState blendAttachmentState =
			vks::initializers::pipelineColorBlendAttachmentState(
				vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA, 
				VK_FALSE);

		vk::PipelineColorBlendStateCreateInfo colorBlendState =
			vks::initializers::pipelineColorBlendStateCreateInfo(1,	&blendAttachmentState);

		vk::PipelineDepthStencilStateCreateInfo depthStencilState =
			vks::initializers::pipelineDepthStencilStateCreateInfo(VK_TRUE, VK_TRUE, vk::CompareOp::eLessOrEqual);

		vk::PipelineViewportStateCreateInfo viewportState =
			vks::initializers::pipelineViewportStateCreateInfo(1, 1);

		vk::PipelineMultisampleStateCreateInfo multisampleState =
			vks::initializers::pipelineMultisampleStateCreateInfo(vk::SampleCountFlagBits::e1);

		std::vector<vk::DynamicState> dynamicStateEnables = {	vk::DynamicState::eViewport, vk::DynamicState::eScissor };
		vk::PipelineDynamicStateCreateInfo dynamicState =
			vks::initializers::pipelineDynamicStateCreateInfo(dynamicStateEnables);

		std::array<vk::PipelineShaderStageCreateInfo, 2> shaderStages;
		
		shaderStages[0] = loadShader(getAssetPath() + "shaders/subpasses/composition.vert.spv", vk::ShaderStageFlagBits::eVertex);
		shaderStages[1] = loadShader(getAssetPath() + "shaders/subpasses/composition.frag.spv", vk::ShaderStageFlagBits::eFragment);

		// Use specialization constants to pass number of lights to the shader
		vk::SpecializationMapEntry specializationEntry{};
		specializationEntry.constantID = 0;
		specializationEntry.offset = 0;
		specializationEntry.size = sizeof(uint32_t);

		uint32_t specializationData = NUM_LIGHTS;

		vk::SpecializationInfo specializationInfo;
		specializationInfo.mapEntryCount = 1;
		specializationInfo.pMapEntries = &specializationEntry;
		specializationInfo.dataSize = sizeof(specializationData);
		specializationInfo.pData = &specializationData;

		shaderStages[1].pSpecializationInfo = &specializationInfo;

		vk::GraphicsPipelineCreateInfo pipelineCreateInfo =
			vks::initializers::pipelineCreateInfo(pipelineLayouts.composition, renderPass);

		vk::PipelineVertexInputStateCreateInfo emptyInputState{};


		pipelineCreateInfo.pVertexInputState = &emptyInputState;
		pipelineCreateInfo.pInputAssemblyState = &inputAssemblyState;
		pipelineCreateInfo.pRasterizationState = &rasterizationState;
		pipelineCreateInfo.pColorBlendState = &colorBlendState;
		pipelineCreateInfo.pMultisampleState = &multisampleState;
		pipelineCreateInfo.pViewportState = &viewportState;
		pipelineCreateInfo.pDepthStencilState = &depthStencilState;
		pipelineCreateInfo.pDynamicState = &dynamicState;
		pipelineCreateInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
		pipelineCreateInfo.pStages = shaderStages.data();
		// Index of the subpass that this pipeline will be used in
		pipelineCreateInfo.subpass = 1;

		depthStencilState.depthWriteEnable = VK_FALSE;

		pipelines.composition = device.createGraphicsPipelines(pipelineCache, pipelineCreateInfo)[0];

		// Transparent (forward) pipeline

		// Descriptor set layout
		setLayoutBindings = {
			vks::initializers::descriptorSetLayoutBinding(vk::DescriptorType::eUniformBuffer, vk::ShaderStageFlagBits::eVertex, 0),
			vks::initializers::descriptorSetLayoutBinding(vk::DescriptorType::eInputAttachment, vk::ShaderStageFlagBits::eFragment, 1),
			vks::initializers::descriptorSetLayoutBinding(vk::DescriptorType::eCombinedImageSampler, vk::ShaderStageFlagBits::eFragment, 2),
		};

		descriptorLayout = vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
		descriptorSetLayouts.transparent = device.createDescriptorSetLayout(descriptorLayout);

		// Pipeline layout
		pPipelineLayoutCreateInfo = vks::initializers::pipelineLayoutCreateInfo(&descriptorSetLayouts.transparent, 1);
		pipelineLayouts.transparent = device.createPipelineLayout(pPipelineLayoutCreateInfo);

		// Descriptor sets
		allocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorPool, &descriptorSetLayouts.transparent, 1);
		descriptorSets.transparent = device.allocateDescriptorSets(allocInfo)[0];

		writeDescriptorSets = {
			vks::initializers::writeDescriptorSet(descriptorSets.transparent, vk::DescriptorType::eUniformBuffer, 0, &uniformBuffers.GBuffer.descriptor),
			vks::initializers::writeDescriptorSet(descriptorSets.transparent, vk::DescriptorType::eInputAttachment, 1, &texDescriptorPosition),
			vks::initializers::writeDescriptorSet(descriptorSets.transparent, vk::DescriptorType::eCombinedImageSampler, 2, &textures.glass.descriptor),
		};
		device.updateDescriptorSets(writeDescriptorSets, nullptr);

		// Enable blending
		blendAttachmentState.blendEnable = VK_TRUE;
		blendAttachmentState.srcColorBlendFactor = vk::BlendFactor::eSrcAlpha;
		blendAttachmentState.dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha;
		blendAttachmentState.colorBlendOp = vk::BlendOp::eAdd;
		blendAttachmentState.srcAlphaBlendFactor = vk::BlendFactor::eOne;
		blendAttachmentState.dstAlphaBlendFactor = vk::BlendFactor::eZero;
		blendAttachmentState.alphaBlendOp = vk::BlendOp::eAdd;
		blendAttachmentState.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;

		pipelineCreateInfo.pVertexInputState = &vertices.inputState;
		pipelineCreateInfo.layout = pipelineLayouts.transparent;
		pipelineCreateInfo.subpass = 2;

		shaderStages[0] = loadShader(getAssetPath() + "shaders/subpasses/transparent.vert.spv", vk::ShaderStageFlagBits::eVertex);
		shaderStages[1] = loadShader(getAssetPath() + "shaders/subpasses/transparent.frag.spv", vk::ShaderStageFlagBits::eFragment);

		pipelines.transparent = device.createGraphicsPipelines(pipelineCache, pipelineCreateInfo)[0];
	}

	// Prepare and initialize uniform buffer containing shader uniforms
	void prepareUniformBuffers()
	{
		// Deferred vertex shader
		vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eUniformBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			&uniformBuffers.GBuffer,
			sizeof(uboGBuffer));

		// Deferred fragment shader
		vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eUniformBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			&uniformBuffers.lights,
			sizeof(uboLights));

		// Update
		updateUniformBufferDeferredMatrices();
		updateUniformBufferDeferredLights();
	}

	void updateUniformBufferDeferredMatrices()
	{
		uboGBuffer.projection = camera.matrices.perspective;
		uboGBuffer.view = camera.matrices.view;
		uboGBuffer.model = glm::mat4();

		uniformBuffers.GBuffer.map();
		memcpy(uniformBuffers.GBuffer.mapped, &uboGBuffer, sizeof(uboGBuffer));
		uniformBuffers.GBuffer.unmap();
	}

	void initLights()
	{
		std::vector<glm::vec3> colors =
		{
			glm::vec3(1.0f, 1.0f, 1.0f),
			glm::vec3(1.0f, 0.0f, 0.0f),
			glm::vec3(0.0f, 1.0f, 0.0f),
			glm::vec3(0.0f, 0.0f, 1.0f),
			glm::vec3(1.0f, 1.0f, 0.0f),
		};

		std::mt19937 rndGen((unsigned)time(NULL));
		std::uniform_real_distribution<float> rndDist(-1.0f, 1.0f);
		std::uniform_int_distribution<uint32_t> rndCol(0, static_cast<uint32_t>(colors.size()-1));

		for (auto& light : uboLights.lights)
		{
			light.position = glm::vec4(rndDist(rndGen) * 6.0f, 0.25f + std::abs(rndDist(rndGen)) * 4.0f, rndDist(rndGen) * 6.0f, 1.0f);
			light.color = colors[rndCol(rndGen)];
			light.radius = 1.0f + std::abs(rndDist(rndGen));			
		}
	}

	// Update fragment shader light position uniform block
	void updateUniformBufferDeferredLights()
	{
		// Current view position
		uboLights.viewPos = glm::vec4(camera.position, 0.0f) * glm::vec4(-1.0f, 1.0f, -1.0f, 1.0f);

		uniformBuffers.lights.map();
		memcpy(uniformBuffers.lights.mapped, &uboLights, sizeof(uboLights));
		uniformBuffers.lights.unmap();
	}

	void draw()
	{
		VulkanExampleBase::prepareFrame();

		// Command buffer to be sumitted to the queue
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &drawCmdBuffers[currentBuffer];

		// Submit to queue
		queue.submit(submitInfo, vk::Fence(nullptr));

		VulkanExampleBase::submitFrame();
	}

	void prepare()
	{
		VulkanExampleBase::prepare();
		loadAssets();
		setupVertexDescriptions();
		initLights();
		prepareUniformBuffers();
		setupDescriptorSetLayout();
		preparePipelines();
		setupDescriptorPool();
		setupDescriptorSet();
		prepareCompositionPass();
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
		updateUniformBufferDeferredMatrices();
		updateUniformBufferDeferredLights();
	}

	virtual void keyPressed(uint32_t keyCode)
	{
		switch (keyCode)
		{
		case KEY_F1:
		case GAMEPAD_BUTTON_A:
			initLights();
			updateUniformBufferDeferredLights();
			break;
		}
	}
};

VULKAN_EXAMPLE_MAIN()
