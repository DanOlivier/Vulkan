/*
* Vulkan Example - Implements a separable two-pass fullscreen blur (also known as bloom)
*
* Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include "VulkanExampleBase.hpp"
#include "VulkanTexture.hpp"
#include "VulkanModel.hpp"

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
		uint32_t width, height;
		vk::RenderPass renderPass;
		vk::Sampler sampler;
		vk::CommandBuffer commandBuffer;
		// Semaphore used to synchronize between offscreen and final scene rendering
		vk::Semaphore semaphore;
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

		device.destroySampler(offscreenPass.sampler);

		// Frame buffer
		for (auto& framebuffer : offscreenPass.framebuffers)
		{
			// Attachments
			device.destroyImageView(framebuffer.color.view);
			device.destroyImage(framebuffer.color.image);
			device.freeMemory(framebuffer.color.mem);
			device.destroyImageView(framebuffer.depth.view);
			device.destroyImage(framebuffer.depth.image);
			device.freeMemory(framebuffer.depth.mem);

			device.destroyFramebuffer(framebuffer.framebuffer);
		}
		device.destroyRenderPass(offscreenPass.renderPass);
		device.freeCommandBuffers(cmdPool, offscreenPass.commandBuffer);
		device.destroySemaphore(offscreenPass.semaphore);

		device.destroyPipeline(pipelines.blurHorz);
		device.destroyPipeline(pipelines.blurVert);
		device.destroyPipeline(pipelines.phongPass);
		device.destroyPipeline(pipelines.glowPass);
		device.destroyPipeline(pipelines.skyBox);

		device.destroyPipelineLayout(pipelineLayouts.blur );
		device.destroyPipelineLayout(pipelineLayouts.scene);

		device.destroyDescriptorSetLayout(descriptorSetLayouts.blur);
		device.destroyDescriptorSetLayout(descriptorSetLayouts.scene);

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
		vk::ImageCreateInfo image;
		image.imageType = vk::ImageType::e2D;
		image.format = colorFormat;
		image.extent = vk::Extent3D{ FB_DIM, FB_DIM, 1 };
		image.mipLevels = 1;
		image.arrayLayers = 1;
		image.samples = vk::SampleCountFlagBits::e1;
		image.tiling = vk::ImageTiling::eOptimal;
		// We will sample directly from the color attachment
		image.usage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled;

		vk::MemoryAllocateInfo memAlloc;
		vk::MemoryRequirements memReqs;

		vk::ImageViewCreateInfo colorImageView;
		colorImageView.viewType = vk::ImageViewType::e2D;
		colorImageView.format = colorFormat;
		//colorImageView.flags = 0;
		colorImageView.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
		colorImageView.subresourceRange.baseMipLevel = 0;
		colorImageView.subresourceRange.levelCount = 1;
		colorImageView.subresourceRange.baseArrayLayer = 0;
		colorImageView.subresourceRange.layerCount = 1;

		frameBuf->color.image = device.createImage(image);
		memReqs = device.getImageMemoryRequirements(frameBuf->color.image);
		memAlloc.allocationSize = memReqs.size;
		memAlloc.memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);
		frameBuf->color.mem = device.allocateMemory(memAlloc);
		device.bindImageMemory(frameBuf->color.image, frameBuf->color.mem, 0);

		colorImageView.image = frameBuf->color.image;
		frameBuf->color.view = device.createImageView(colorImageView);

		// Depth stencil attachment
		image.format = depthFormat;
		image.usage = vk::ImageUsageFlagBits::eDepthStencilAttachment;

		vk::ImageViewCreateInfo depthStencilView;
		depthStencilView.viewType = vk::ImageViewType::e2D;
		depthStencilView.format = depthFormat;
		//depthStencilView.flags = 0;
		//depthStencilView.subresourceRange = {};
		depthStencilView.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil;
		depthStencilView.subresourceRange.baseMipLevel = 0;
		depthStencilView.subresourceRange.levelCount = 1;
		depthStencilView.subresourceRange.baseArrayLayer = 0;
		depthStencilView.subresourceRange.layerCount = 1;

		frameBuf->depth.image = device.createImage(image);
		memReqs = device.getImageMemoryRequirements(frameBuf->depth.image);
		memAlloc.allocationSize = memReqs.size;
		memAlloc.memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);
		frameBuf->depth.mem = device.allocateMemory(memAlloc);
		device.bindImageMemory(frameBuf->depth.image, frameBuf->depth.mem, 0);

		depthStencilView.image = frameBuf->depth.image;
		frameBuf->depth.view = device.createImageView(depthStencilView);

		vk::ImageView attachments[2];
		attachments[0] = frameBuf->color.view;
		attachments[1] = frameBuf->depth.view;

		vk::FramebufferCreateInfo fbufCreateInfo;
		fbufCreateInfo.renderPass = offscreenPass.renderPass;
		fbufCreateInfo.attachmentCount = 2;
		fbufCreateInfo.pAttachments = attachments;
		fbufCreateInfo.width = FB_DIM;
		fbufCreateInfo.height = FB_DIM;
		fbufCreateInfo.layers = 1;

		frameBuf->framebuffer = device.createFramebuffer(fbufCreateInfo);

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

		renderPassInfo.attachmentCount = static_cast<uint32_t>(attchmentDescriptions.size());
		renderPassInfo.pAttachments = attchmentDescriptions.data();
		renderPassInfo.subpassCount = 1;
		renderPassInfo.pSubpasses = &subpassDescription;
		renderPassInfo.dependencyCount = static_cast<uint32_t>(dependencies.size());
		renderPassInfo.pDependencies = dependencies.data();

		offscreenPass.renderPass = device.createRenderPass(renderPassInfo);

		// Create sampler to sample from the color attachments
		vk::SamplerCreateInfo sampler;
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
		offscreenPass.sampler = device.createSampler(sampler);

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
		if (!offscreenPass.commandBuffer)
		{
			offscreenPass.commandBuffer = VulkanExampleBase::createCommandBuffer(vk::CommandBufferLevel::ePrimary, false);
		}

		if (!offscreenPass.semaphore)
		{
			vk::SemaphoreCreateInfo semaphoreCreateInfo;
			offscreenPass.semaphore = device.createSemaphore(semaphoreCreateInfo);
		}

		vk::CommandBufferBeginInfo cmdBufInfo;

		// First pass: Render glow parts of the model (separate mesh)
		// -------------------------------------------------------------------------------------------------------

		vk::ClearValue clearValues[2];
		clearValues[0].color = vk::ClearColorValue{ std::array<float, 4>{ 0.0f, 0.0f, 0.0f, 1.0f } };
		clearValues[1].depthStencil = vk::ClearDepthStencilValue{ 1.0f, 0 };

		vk::RenderPassBeginInfo renderPassBeginInfo;
		renderPassBeginInfo.renderPass = offscreenPass.renderPass;
		renderPassBeginInfo.framebuffer = offscreenPass.framebuffers[0].framebuffer;
		renderPassBeginInfo.renderArea.extent.width = offscreenPass.width;
		renderPassBeginInfo.renderArea.extent.height = offscreenPass.height;
		renderPassBeginInfo.clearValueCount = 2;
		renderPassBeginInfo.pClearValues = clearValues;

		offscreenPass.commandBuffer.begin(cmdBufInfo);

		vk::Viewport viewport = vks::initializers::viewport((float)offscreenPass.width, (float)offscreenPass.height, 0.0f, 1.0f);
		offscreenPass.commandBuffer.setViewport(0, viewport);

		vk::Rect2D scissor = vks::initializers::rect2D(offscreenPass.width, offscreenPass.height,	0, 0);
		offscreenPass.commandBuffer.setScissor(0, scissor);

		offscreenPass.commandBuffer.beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);

		offscreenPass.commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayouts.scene, 0, descriptorSets.scene, nullptr);
		offscreenPass.commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.glowPass);

		std::vector<vk::DeviceSize> offsets = { 0 };
		offscreenPass.commandBuffer.bindVertexBuffers(VERTEX_BUFFER_BIND_ID, models.ufoGlow.vertices.buffer, offsets);
		offscreenPass.commandBuffer.bindIndexBuffer(models.ufoGlow.indices.buffer, 0, vk::IndexType::eUint32);
		offscreenPass.commandBuffer.drawIndexed(models.ufoGlow.indexCount, 1, 0, 0, 0);

		offscreenPass.commandBuffer.endRenderPass();

		// Second pass: Render contents of the first pass into second framebuffer and apply a vertical blur
		// This is the first blur pass, the horizontal blur is applied when rendering on top of the scene
		// -------------------------------------------------------------------------------------------------------

		renderPassBeginInfo.framebuffer = offscreenPass.framebuffers[1].framebuffer;

		offscreenPass.commandBuffer.beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);

		offscreenPass.commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayouts.blur, 0, descriptorSets.blurVert, nullptr);
		offscreenPass.commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.blurVert);
		vkCmdDraw(offscreenPass.commandBuffer, 3, 1, 0, 0);

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
		vk::CommandBufferBeginInfo cmdBufInfo;

		vk::ClearValue clearValues[2];
		clearValues[0].color = defaultClearColor;
		clearValues[1].depthStencil = vk::ClearDepthStencilValue{ 1.0f, 0 };

		vk::RenderPassBeginInfo renderPassBeginInfo;
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

			// Skybox 
			drawCmdBuffers[i].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayouts.scene, 0, descriptorSets.skyBox, nullptr);
			drawCmdBuffers[i].bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.skyBox);

			drawCmdBuffers[i].bindVertexBuffers(VERTEX_BUFFER_BIND_ID, models.skyBox.vertices.buffer, offsets);
			drawCmdBuffers[i].bindIndexBuffer(models.skyBox.indices.buffer, 0, vk::IndexType::eUint32);
			drawCmdBuffers[i].drawIndexed(models.skyBox.indexCount, 1, 0, 0, 0);

			// 3D scene
			drawCmdBuffers[i].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayouts.scene, 0, descriptorSets.scene, nullptr);
			drawCmdBuffers[i].bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.phongPass);

			drawCmdBuffers[i].bindVertexBuffers(VERTEX_BUFFER_BIND_ID, models.ufo.vertices.buffer, offsets);
			drawCmdBuffers[i].bindIndexBuffer(models.ufo.indices.buffer, 0, vk::IndexType::eUint32);
			drawCmdBuffers[i].drawIndexed(models.ufo.indexCount, 1, 0, 0, 0);

			// Render vertical blurred scene applying a horizontal blur
			// Render the (vertically blurred) contents of the second framebuffer and apply a horizontal blur
			// -------------------------------------------------------------------------------------------------------
			if (bloom)
			{
				drawCmdBuffers[i].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayouts.blur, 0, descriptorSets.blurHorz, nullptr);
				drawCmdBuffers[i].bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.blurHorz);
				vkCmdDraw(drawCmdBuffers[i], 3, 1, 0, 0);
			}

			drawCmdBuffers[i].endRenderPass();

			drawCmdBuffers[i].end();
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

		descriptorPool = device.createDescriptorPool(descriptorPoolInfo);
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
		descriptorSetLayoutCreateInfo = vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
		descriptorSetLayouts.blur = device.createDescriptorSetLayout(descriptorSetLayoutCreateInfo);
		pipelineLayoutCreateInfo = vks::initializers::pipelineLayoutCreateInfo(&descriptorSetLayouts.blur, 1);
		pipelineLayouts.blur = device.createPipelineLayout(pipelineLayoutCreateInfo);

		// Scene rendering
		setLayoutBindings = {			
			vks::initializers::descriptorSetLayoutBinding(vk::DescriptorType::eUniformBuffer, vk::ShaderStageFlagBits::eVertex, 0),			// Binding 0 : Vertex shader uniform buffer			
			vks::initializers::descriptorSetLayoutBinding(vk::DescriptorType::eCombinedImageSampler, vk::ShaderStageFlagBits::eFragment, 1),	// Binding 1 : Fragment shader image sampler			
			vks::initializers::descriptorSetLayoutBinding(vk::DescriptorType::eUniformBuffer, vk::ShaderStageFlagBits::eFragment, 2),			// Binding 2 : Framgnet shader image sampler
		};

		descriptorSetLayoutCreateInfo = vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings.data(), setLayoutBindings.size());
		descriptorSetLayouts.scene = device.createDescriptorSetLayout(descriptorSetLayoutCreateInfo);
		pipelineLayoutCreateInfo = vks::initializers::pipelineLayoutCreateInfo(&descriptorSetLayouts.scene, 1);
		pipelineLayouts.scene = device.createPipelineLayout(pipelineLayoutCreateInfo);
	}

	void setupDescriptorSet()
	{
		vk::DescriptorSetAllocateInfo descriptorSetAllocInfo;
		std::vector<vk::WriteDescriptorSet> writeDescriptorSets;

		// Full screen blur
		// Vertical
		descriptorSetAllocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorPool, &descriptorSetLayouts.blur, 1);
		descriptorSets.blurVert = device.allocateDescriptorSets(descriptorSetAllocInfo)[0];
		writeDescriptorSets = {			
			vks::initializers::writeDescriptorSet(descriptorSets.blurVert, vk::DescriptorType::eUniformBuffer, 0, &uniformBuffers.blurParams.descriptor),				// Binding 0: Fragment shader uniform buffer			
			vks::initializers::writeDescriptorSet(descriptorSets.blurVert, vk::DescriptorType::eCombinedImageSampler, 1, &offscreenPass.framebuffers[0].descriptor),	// Binding 1: Fragment shader texture sampler
		};
		device.updateDescriptorSets(writeDescriptorSets, nullptr);
		// Horizontal
		descriptorSetAllocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorPool, &descriptorSetLayouts.blur, 1);
		descriptorSets.blurHorz = device.allocateDescriptorSets(descriptorSetAllocInfo)[0];
		writeDescriptorSets = {
			vks::initializers::writeDescriptorSet(descriptorSets.blurHorz, vk::DescriptorType::eUniformBuffer, 0, &uniformBuffers.blurParams.descriptor),				// Binding 0: Fragment shader uniform buffer			
			vks::initializers::writeDescriptorSet(descriptorSets.blurHorz, vk::DescriptorType::eCombinedImageSampler, 1, &offscreenPass.framebuffers[1].descriptor),	// Binding 1: Fragment shader texture sampler
		};
		device.updateDescriptorSets(writeDescriptorSets, nullptr);

		// Scene rendering
		descriptorSetAllocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorPool, &descriptorSetLayouts.scene, 1);
		descriptorSets.scene = device.allocateDescriptorSets(descriptorSetAllocInfo)[0];
		writeDescriptorSets = {			
			vks::initializers::writeDescriptorSet(descriptorSets.scene, vk::DescriptorType::eUniformBuffer, 0, &uniformBuffers.scene.descriptor)							// Binding 0: Vertex shader uniform buffer
		};
		device.updateDescriptorSets(writeDescriptorSets, nullptr);

		// Skybox
		descriptorSets.skyBox = device.allocateDescriptorSets(descriptorSetAllocInfo)[0];
		writeDescriptorSets = {			
			vks::initializers::writeDescriptorSet(descriptorSets.skyBox, vk::DescriptorType::eUniformBuffer, 0, &uniformBuffers.skyBox.descriptor),						// Binding 0: Vertex shader uniform buffer			
			vks::initializers::writeDescriptorSet(descriptorSets.skyBox, vk::DescriptorType::eCombinedImageSampler,	1, &textures.cubemap.descriptor),					// Binding 1: Fragment shader texture sampler
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
				vk::FrontFace::eClockwise);

		vk::PipelineColorBlendAttachmentState blendAttachmentState =
			vks::initializers::pipelineColorBlendAttachmentState();

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
			vks::initializers::pipelineMultisampleStateCreateInfo(
				vk::SampleCountFlagBits::e1);

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
				pipelineLayouts.blur,
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

		// Blur pipelines
		shaderStages[0] = loadShader(getAssetPath() + "shaders/gaussblur.vert.spv", vk::ShaderStageFlagBits::eVertex);
		shaderStages[1] = loadShader(getAssetPath() + "shaders/gaussblur.frag.spv", vk::ShaderStageFlagBits::eFragment);
		// Empty vertex input state
		vk::PipelineVertexInputStateCreateInfo emptyInputState = vks::initializers::pipelineVertexInputStateCreateInfo();
		pipelineCreateInfo.pVertexInputState = &emptyInputState;
		pipelineCreateInfo.layout = pipelineLayouts.blur;
		// Additive blending
		blendAttachmentState.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
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
		pipelines.blurVert = device.createGraphicsPipelines(pipelineCache, pipelineCreateInfo)[0];
		// Horizontal blur pipeline
		blurdirection = 1;
		pipelineCreateInfo.renderPass = renderPass;
		pipelines.blurHorz = device.createGraphicsPipelines(pipelineCache, pipelineCreateInfo)[0];

		// Phong pass (3D model)
		pipelineCreateInfo.layout = pipelineLayouts.scene;
		pipelineCreateInfo.pVertexInputState = &vertices.inputState;
		shaderStages[0] = loadShader(getAssetPath() + "shaders/phongpass.vert.spv", vk::ShaderStageFlagBits::eVertex);
		shaderStages[1] = loadShader(getAssetPath() + "shaders/phongpass.frag.spv", vk::ShaderStageFlagBits::eFragment);
		blendAttachmentState.blendEnable = VK_FALSE;
		depthStencilState.depthWriteEnable = VK_TRUE;
		rasterizationState.cullMode = vk::CullModeFlagBits::eBack;
		pipelineCreateInfo.renderPass = renderPass;
		pipelines.phongPass = device.createGraphicsPipelines(pipelineCache, pipelineCreateInfo)[0];

		// Color only pass (offscreen blur base)
		shaderStages[0] = loadShader(getAssetPath() + "shaders/colorpass.vert.spv", vk::ShaderStageFlagBits::eVertex);
		shaderStages[1] = loadShader(getAssetPath() + "shaders/colorpass.frag.spv", vk::ShaderStageFlagBits::eFragment);
		pipelineCreateInfo.renderPass = offscreenPass.renderPass;
		pipelines.glowPass = device.createGraphicsPipelines(pipelineCache, pipelineCreateInfo)[0];

		// Skybox (cubemap)
		shaderStages[0] = loadShader(getAssetPath() + "shaders/skybox.vert.spv", vk::ShaderStageFlagBits::eVertex);
		shaderStages[1] = loadShader(getAssetPath() + "shaders/skybox.frag.spv", vk::ShaderStageFlagBits::eFragment);
		depthStencilState.depthWriteEnable = VK_FALSE;
		rasterizationState.cullMode = vk::CullModeFlagBits::eFront;
		pipelineCreateInfo.renderPass = renderPass;
		pipelines.skyBox = device.createGraphicsPipelines(pipelineCache, pipelineCreateInfo)[0];
	}

	// Prepare and initialize uniform buffer containing shader uniforms
	void prepareUniformBuffers()
	{
		// Phong and color pass vertex shader uniform buffer
		vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eUniformBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			&uniformBuffers.scene,
			sizeof(ubos.scene));

		// Blur parameters uniform buffers
		vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eUniformBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			&uniformBuffers.blurParams,
			sizeof(ubos.blurParams));

		// Skybox
		vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eUniformBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			&uniformBuffers.skyBox,
			sizeof(ubos.skyBox));

		// Map persistent
		uniformBuffers.scene.map();
		uniformBuffers.blurParams.map();
		uniformBuffers.skyBox.map();

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
