/*
* Vulkan Example - Multisampling using resolve attachments
*
* Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include <stdio.h>
#include <stdlib.h>
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
#include "VulkanTexture.hpp"
#include "VulkanModel.hpp"

#define VERTEX_BUFFER_BIND_ID 0
#define ENABLE_VALIDATION false

struct {
	struct {
		vk::Image image;
		vk::ImageView view;
		vk::DeviceMemory memory;
	} color;
	struct {
		vk::Image image;
		vk::ImageView view;
		vk::DeviceMemory memory;
	} depth;
} multisampleTarget;

class VulkanExample : public VulkanExampleBase
{
public:
	bool useSampleShading = false;
    vk::SampleCountFlagBits sampleCount = vk::SampleCountFlagBits::e1;

	struct {
		vks::Texture2D colorMap;
	} textures;

	// Vertex layout for the models
	vks::VertexLayout vertexLayout = vks::VertexLayout({
		vks::VERTEX_COMPONENT_POSITION,
		vks::VERTEX_COMPONENT_NORMAL,
		vks::VERTEX_COMPONENT_UV,
		vks::VERTEX_COMPONENT_COLOR,
	});

	struct {
		vks::Model example;
	} models;

	struct {
		vk::PipelineVertexInputStateCreateInfo inputState;
		std::vector<vk::VertexInputBindingDescription> bindingDescriptions;
		std::vector<vk::VertexInputAttributeDescription> attributeDescriptions;
	} vertices;

	vks::Buffer uniformBuffer;

	struct UBOVS {
		glm::mat4 projection;
		glm::mat4 model;
		glm::vec4 lightPos = glm::vec4(5.0f, 5.0f, 5.0f, 1.0f);
	} uboVS;

	struct {
		vk::Pipeline MSAA;
		vk::Pipeline MSAASampleShading;
	} pipelines;

	vk::PipelineLayout pipelineLayout;
	vk::DescriptorSet descriptorSet;
	vk::DescriptorSetLayout descriptorSetLayout;

	VulkanExample() : VulkanExampleBase(ENABLE_VALIDATION)
	{
		zoom = -7.5f;
		zoomSpeed = 2.5f;
		rotation = { 0.0f, -90.0f, 0.0f };
		cameraPos = glm::vec3(2.5f, 2.5f, 0.0f);
		title = "Vulkan Example - Multisampling";
	}

	~VulkanExample()
	{
		// Clean up used Vulkan resources 
		// Note : Inherited destructor cleans up resources stored in base class
		device.destroyPipeline(pipelines.MSAA);
		device.destroyPipeline(pipelines.MSAASampleShading);

		device.destroyPipelineLayout(pipelineLayout);
		device.destroyDescriptorSetLayout(descriptorSetLayout);

		models.example.destroy();

		// Destroy MSAA target
		device.destroyImage(multisampleTarget.color.image);
		device.destroyImageView(multisampleTarget.color.view);
		device.freeMemory(multisampleTarget.color.memory);
		device.destroyImage(multisampleTarget.depth.image);
		device.destroyImageView(multisampleTarget.depth.view);
		device.freeMemory(multisampleTarget.depth.memory);

		textures.colorMap.destroy();

		uniformBuffer.destroy();
	}

	// Creates a multi sample render target (image and view) that is used to resolve 
	// into the visible frame buffer target in the render pass
	void setupMultisampleTarget()
	{
		// Check if device supports requested sample count for color and depth frame buffer
		assert((deviceProperties.limits.framebufferColorSampleCounts >= sampleCount) && 
			(deviceProperties.limits.framebufferDepthSampleCounts >= sampleCount));

		// Color target
		vk::ImageCreateInfo info = vks::initializers::imageCreateInfo();
		info.imageType = vk::ImageType::e2D;
		info.format = swapChain.colorFormat;
		info.extent.width = width;
		info.extent.height = height;
		info.extent.depth = 1;
		info.mipLevels = 1;
		info.arrayLayers = 1;
		info.sharingMode = vk::SharingMode::eExclusive;
		info.tiling = vk::ImageTiling::eOptimal;
		info.samples = sampleCount;
		// Image will only be used as a transient target
		info.usage = vk::ImageUsageFlagBits::eTransientAttachment | vk::ImageUsageFlagBits::eColorAttachment;
		info.initialLayout = vk::ImageLayout::eUndefined;

		multisampleTarget.color.image = device.createImage(info);

		vk::MemoryRequirements memReqs;
		memReqs = device.getImageMemoryRequirements(multisampleTarget.color.image);
		vk::MemoryAllocateInfo memAlloc = vks::initializers::memoryAllocateInfo();
		memAlloc.allocationSize = memReqs.size;
		// We prefer a lazily allocated memory type
		// This means that the memory gets allocated when the implementation sees fit, e.g. when first using the images
		vk::Bool32 lazyMemTypePresent;
		memAlloc.memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eLazilyAllocated, &lazyMemTypePresent);
		if (!lazyMemTypePresent)
		{
			// If this is not available, fall back to device local memory
			memAlloc.memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);
		}
		multisampleTarget.color.memory = device.allocateMemory(memAlloc);
		vkBindImageMemory(device, multisampleTarget.color.image, multisampleTarget.color.memory, 0);

		// Create image view for the MSAA target
		vk::ImageViewCreateInfo viewInfo = vks::initializers::imageViewCreateInfo();
		viewInfo.image = multisampleTarget.color.image;
		viewInfo.viewType = vk::ImageViewType::e2D;
		viewInfo.format = swapChain.colorFormat;
		viewInfo.components.r = vk::ComponentSwizzle::eR;
		viewInfo.components.g = vk::ComponentSwizzle::eG;
		viewInfo.components.b = vk::ComponentSwizzle::eB;
		viewInfo.components.a = vk::ComponentSwizzle::eA;
		viewInfo.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
		viewInfo.subresourceRange.levelCount = 1;
		viewInfo.subresourceRange.layerCount = 1;

		multisampleTarget.color.view = device.createImageView(viewInfo);

		// Depth target
		info.imageType = vk::ImageType::e2D;
		info.format = depthFormat;
		info.extent.width = width;
		info.extent.height = height;
		info.extent.depth = 1;
		info.mipLevels = 1;
		info.arrayLayers = 1;
		info.sharingMode = vk::SharingMode::eExclusive;
		info.tiling = vk::ImageTiling::eOptimal;
		info.samples = sampleCount;
		// Image will only be used as a transient target
		info.usage = vk::ImageUsageFlagBits::eTransientAttachment | vk::ImageUsageFlagBits::eDepthStencilAttachment;
		info.initialLayout = vk::ImageLayout::eUndefined;

		multisampleTarget.depth.image = device.createImage(info);

		memReqs = device.getImageMemoryRequirements(multisampleTarget.depth.image);
		memAlloc = vks::initializers::memoryAllocateInfo();
		memAlloc.allocationSize = memReqs.size;

		memAlloc.memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eLazilyAllocated, &lazyMemTypePresent);
		if (!lazyMemTypePresent)
		{
			memAlloc.memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);
		}
		
		multisampleTarget.depth.memory = device.allocateMemory(memAlloc);
		vkBindImageMemory(device, multisampleTarget.depth.image, multisampleTarget.depth.memory, 0);

		// Create image view for the MSAA target
		viewInfo.image = multisampleTarget.depth.image;
		viewInfo.viewType = vk::ImageViewType::e2D;
		viewInfo.format = depthFormat;
		viewInfo.components.r = vk::ComponentSwizzle::eR;
		viewInfo.components.g = vk::ComponentSwizzle::eG;
		viewInfo.components.b = vk::ComponentSwizzle::eB;
		viewInfo.components.a = vk::ComponentSwizzle::eA;
		viewInfo.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil;
		viewInfo.subresourceRange.levelCount = 1;
		viewInfo.subresourceRange.layerCount = 1;

		multisampleTarget.depth.view = device.createImageView(viewInfo);
	}

	// Setup a render pass for using a multi sampled attachment 
	// and a resolve attachment that the msaa image is resolved 
	// to at the end of the render pass
	void setupRenderPass()
	{
		// Overrides the virtual function of the base class

		std::array<vk::AttachmentDescription, 4> attachments = {};

		// Multisampled attachment that we render to
		attachments[0].format = swapChain.colorFormat;
		attachments[0].samples = sampleCount;
		attachments[0].loadOp = vk::AttachmentLoadOp::eClear;
		// No longer required after resolve, this may save some bandwidth on certain GPUs
		attachments[0].storeOp = vk::AttachmentStoreOp::eDontCare;
		attachments[0].stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
		attachments[0].stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
		attachments[0].initialLayout = vk::ImageLayout::eUndefined;
		attachments[0].finalLayout = vk::ImageLayout::eColorAttachmentOptimal;

		// This is the frame buffer attachment to where the multisampled image
		// will be resolved to and which will be presented to the swapchain
		attachments[1].format = swapChain.colorFormat;
		attachments[1].samples = vk::SampleCountFlagBits::e1;
		attachments[1].loadOp = vk::AttachmentLoadOp::eDontCare;
		attachments[1].storeOp = vk::AttachmentStoreOp::eStore;
		attachments[1].stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
		attachments[1].stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
		attachments[1].initialLayout = vk::ImageLayout::eUndefined;
		attachments[1].finalLayout = vk::ImageLayout::ePresentSrcKHR;

		// Multisampled depth attachment we render to
		attachments[2].format = depthFormat;
		attachments[2].samples = sampleCount;
		attachments[2].loadOp = vk::AttachmentLoadOp::eClear;
		attachments[2].storeOp = vk::AttachmentStoreOp::eDontCare;
		attachments[2].stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
		attachments[2].stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
		attachments[2].initialLayout = vk::ImageLayout::eUndefined;
		attachments[2].finalLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal;

		// Depth resolve attachment
		attachments[3].format = depthFormat;
		attachments[3].samples = vk::SampleCountFlagBits::e1;
		attachments[3].loadOp = vk::AttachmentLoadOp::eDontCare;
		attachments[3].storeOp = vk::AttachmentStoreOp::eStore;
		attachments[3].stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
		attachments[3].stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
		attachments[3].initialLayout = vk::ImageLayout::eUndefined;
		attachments[3].finalLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal;

		vk::AttachmentReference colorReference = {};
		colorReference.attachment = 0;
		colorReference.layout = vk::ImageLayout::eColorAttachmentOptimal;

		vk::AttachmentReference depthReference = {};
		depthReference.attachment = 2;
		depthReference.layout = vk::ImageLayout::eDepthStencilAttachmentOptimal;

		// Resolve attachment reference for the color attachment
		vk::AttachmentReference resolveReference = {};
		resolveReference.attachment = 1;
		resolveReference.layout = vk::ImageLayout::eColorAttachmentOptimal;

		vk::SubpassDescription subpass = {};
		subpass.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
		subpass.colorAttachmentCount = 1;
		subpass.pColorAttachments = &colorReference;
		// Pass our resolve attachments to the sub pass
		subpass.pResolveAttachments = &resolveReference;
		subpass.pDepthStencilAttachment = &depthReference;

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

		vk::RenderPassCreateInfo renderPassInfo = vks::initializers::renderPassCreateInfo();
		renderPassInfo.attachmentCount = attachments.size();
		renderPassInfo.pAttachments = attachments.data();
		renderPassInfo.subpassCount = 1;
		renderPassInfo.pSubpasses = &subpass;
		renderPassInfo.dependencyCount = 2;
		renderPassInfo.pDependencies = dependencies.data();

		renderPass = device.createRenderPass(renderPassInfo);
	}

	// Frame buffer attachments must match with render pass setup, 
	// so we need to adjust frame buffer creation to cover our 
	// multisample target
	void setupFrameBuffer()
	{
		// Overrides the virtual function of the base class

		std::array<vk::ImageView, 4> attachments;

		setupMultisampleTarget();

		attachments[0] = multisampleTarget.color.view;
		// attachment[1] = swapchain image
		attachments[2] = multisampleTarget.depth.view;
		attachments[3] = depthStencil.view;

		vk::FramebufferCreateInfo frameBufferCreateInfo = {};
		frameBufferCreateInfo.renderPass = renderPass;
		frameBufferCreateInfo.attachmentCount = attachments.size();
		frameBufferCreateInfo.pAttachments = attachments.data();
		frameBufferCreateInfo.width = width;
		frameBufferCreateInfo.height = height;
		frameBufferCreateInfo.layers = 1;

		// Create frame buffers for every swap chain image
		frameBuffers.resize(swapChain.imageCount);
		for (uint32_t i = 0; i < frameBuffers.size(); i++)
		{
			attachments[1] = swapChain.buffers[i].view;
			frameBuffers[i] = device.createFramebuffer(frameBufferCreateInfo);
		}
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

		vk::ClearValue clearValues[3];
		// Clear to a white background for higher contrast
		clearValues[0].color = vk::ClearColorValue{ std::array<float, 4>{ 1.0f, 1.0f, 1.0f, 1.0f } };
		clearValues[1].color = { { 1.0f, 1.0f, 1.0f, 1.0f } };
		clearValues[2].depthStencil = vk::ClearDepthStencilValue{ 1.0f, 0 };

		vk::RenderPassBeginInfo renderPassBeginInfo = vks::initializers::renderPassBeginInfo();
		renderPassBeginInfo.renderPass = renderPass;
		renderPassBeginInfo.renderArea.extent.width = width;
		renderPassBeginInfo.renderArea.extent.height = height;
		renderPassBeginInfo.clearValueCount = 3;
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

			drawCmdBuffers[i].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSet, nullptr);
			drawCmdBuffers[i].bindPipeline(vk::PipelineBindPoint::eGraphics, useSampleShading ? pipelines.MSAASampleShading : pipelines.MSAA);

			std::vector<vk::DeviceSize> offsets = { 0 };
			drawCmdBuffers[i].bindVertexBuffers(VERTEX_BUFFER_BIND_ID, models.example.vertices.buffer, offsets);
			drawCmdBuffers[i].bindIndexBuffer(models.example.indices.buffer, 0, vk::IndexType::eUint32);
			drawCmdBuffers[i].drawIndexed(models.example.indexCount, 1, 0, 0, 0);

			drawCmdBuffers[i].endRenderPass();

			drawCmdBuffers[i].end();
		}
	}

	void loadAssets()
	{
		models.example.loadFromFile(getAssetPath() + "models/voyager/voyager.dae", vertexLayout, 1.0f, vulkanDevice, queue);
		if (deviceFeatures.textureCompressionBC) {
			textures.colorMap.loadFromFile(getAssetPath() + "models/voyager/voyager_bc3_unorm.ktx", vk::Format::eBc3UnormBlock, vulkanDevice, queue);
		}
		else if (deviceFeatures.textureCompressionASTC_LDR) {
			textures.colorMap.loadFromFile(getAssetPath() + "models/voyager/voyager_astc_8x8_unorm.ktx", vk::Format::eAstc8x8UnormBlock, vulkanDevice, queue);
		}
		else if (deviceFeatures.textureCompressionETC2) {
			textures.colorMap.loadFromFile(getAssetPath() + "models/voyager/voyager_etc2_unorm.ktx", vk::Format::eEtc2R8G8B8A8UnormBlock, vulkanDevice, queue);
		}
		else {
			vks::tools::exitFatal("Device does not support any compressed texture format!", "Error");
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
		// Location 0 : Position
		vertices.attributeDescriptions[0] =
			vks::initializers::vertexInputAttributeDescription(
				VERTEX_BUFFER_BIND_ID,
				0,
				vk::Format::eR32G32B32Sfloat,
				0);
		// Location 1 : Normal
		vertices.attributeDescriptions[1] =
			vks::initializers::vertexInputAttributeDescription(
				VERTEX_BUFFER_BIND_ID,
				1,
				vk::Format::eR32G32B32Sfloat,
				sizeof(float) * 3);
		// Location 2 : Texture coordinates
		vertices.attributeDescriptions[2] =
			vks::initializers::vertexInputAttributeDescription(
				VERTEX_BUFFER_BIND_ID,
				2,
				vk::Format::eR32G32Sfloat,
				sizeof(float) * 6);
		// Location 3 : Color
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
		// Example uses one ubo and one combined image sampler
		std::vector<vk::DescriptorPoolSize> poolSizes =
		{
			vks::initializers::descriptorPoolSize(vk::DescriptorType::eUniformBuffer, 1),
			vks::initializers::descriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, 1),
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
		std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings =
		{
			// Binding 0 : Vertex shader uniform buffer
			vks::initializers::descriptorSetLayoutBinding(
			vk::DescriptorType::eUniformBuffer,
				vk::ShaderStageFlagBits::eVertex,
				0),
			// Binding 1 : Fragment shader combined sampler
			vks::initializers::descriptorSetLayoutBinding(
				vk::DescriptorType::eCombinedImageSampler,
				vk::ShaderStageFlagBits::eFragment,
				1),
		};

		vk::DescriptorSetLayoutCreateInfo descriptorLayout =
			vks::initializers::descriptorSetLayoutCreateInfo(
				setLayoutBindings.data(),
				setLayoutBindings.size());

		descriptorSetLayout = device.createDescriptorSetLayout(descriptorLayout);

		vk::PipelineLayoutCreateInfo pPipelineLayoutCreateInfo =
			vks::initializers::pipelineLayoutCreateInfo(
				&descriptorSetLayout,
				1);

		pipelineLayout = device.createPipelineLayout(pPipelineLayoutCreateInfo);
	}

	void setupDescriptorSet()
	{
		vk::DescriptorSetAllocateInfo allocInfo =
			vks::initializers::descriptorSetAllocateInfo(
				descriptorPool,
				&descriptorSetLayout,
				1);

		descriptorSet = device.allocateDescriptorSets(allocInfo)[0];
		
		vk::DescriptorImageInfo texDescriptor =
			vks::initializers::descriptorImageInfo(
				textures.colorMap.sampler,
				textures.colorMap.view,
				vk::ImageLayout::eGeneral);

		std::vector<vk::WriteDescriptorSet> writeDescriptorSets =
		{
			// Binding 0 : Vertex shader uniform buffer
			vks::initializers::writeDescriptorSet(
			descriptorSet,
				vk::DescriptorType::eUniformBuffer,
				0,
				&uniformBuffer.descriptor),
			// Binding 1 : Color map 
			vks::initializers::writeDescriptorSet(
				descriptorSet,
				vk::DescriptorType::eCombinedImageSampler,
				1,
				&texDescriptor)
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

		std::vector<vk::DynamicState> dynamicStateEnables = {
			vk::DynamicState::eViewport,
			vk::DynamicState::eScissor
		};
		vk::PipelineDynamicStateCreateInfo dynamicState =
			vks::initializers::pipelineDynamicStateCreateInfo(
				dynamicStateEnables);

		vk::GraphicsPipelineCreateInfo pipelineCreateInfo =
			vks::initializers::pipelineCreateInfo(
				pipelineLayout,
				renderPass);

		vk::PipelineMultisampleStateCreateInfo multisampleState{};


		std::array<vk::PipelineShaderStageCreateInfo, 2> shaderStages;

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

		// MSAA rendering pipeline
		shaderStages[0] = loadShader(getAssetPath() + "shaders/mesh/mesh.vert.spv", vk::ShaderStageFlagBits::eVertex);
		shaderStages[1] = loadShader(getAssetPath() + "shaders/mesh/mesh.frag.spv", vk::ShaderStageFlagBits::eFragment);
		// Setup multi sampling
		multisampleState.rasterizationSamples = sampleCount;		// Number of samples to use for rasterization
		
		pipelines.MSAA = device.createGraphicsPipelines(pipelineCache, pipelineCreateInfo)[0];

		// MSAA with sample shading pipeline
		// Sample shading enables per-sample shading to avoid shader aliasing and smooth out e.g. high frequency texture maps
		// Note: This will trade performance for are more stable image
		multisampleState.sampleShadingEnable = VK_TRUE;				// Enable per-sample shading (instead of per-fragment)
		multisampleState.minSampleShading = 0.25f;					// Minimum fraction for sample shading
		
		pipelines.MSAASampleShading = device.createGraphicsPipelines(pipelineCache, pipelineCreateInfo)[0];
	}

	// Prepare and initialize uniform buffer containing shader uniforms
	void prepareUniformBuffers()
	{
		// Vertex shader uniform buffer block
		vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eUniformBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			&uniformBuffer,
			sizeof(uboVS));

		// Map persistent
		uniformBuffer.map();

		updateUniformBuffers();
	}

	void updateUniformBuffers()
	{
		// Vertex shader
		glm::mat4 viewMatrix = glm::mat4();
		uboVS.projection = glm::perspective(glm::radians(60.0f), (float)width / (float)height, 0.1f, 256.0f);
		viewMatrix = glm::translate(viewMatrix, glm::vec3(0.0f, 0.0f, zoom));

		//float offset = 0.5f;
		//int uboIndex = 1;
		uboVS.model = glm::mat4();
		uboVS.model = viewMatrix * glm::translate(uboVS.model, cameraPos);
		uboVS.model = glm::rotate(uboVS.model, glm::radians(rotation.x), glm::vec3(1.0f, 0.0f, 0.0f));
		uboVS.model = glm::rotate(uboVS.model, glm::radians(rotation.y), glm::vec3(0.0f, 1.0f, 0.0f));
		uboVS.model = glm::rotate(uboVS.model, glm::radians(rotation.z), glm::vec3(0.0f, 0.0f, 1.0f));

		memcpy(uniformBuffer.mapped, &uboVS, sizeof(uboVS));
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
        sampleCount = getMaxUsableSampleCount();
		VulkanExampleBase::prepare();
		loadAssets();
		setupVertexDescriptions();
		prepareUniformBuffers();
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
	}

	virtual void viewChanged()
	{
		updateUniformBuffers();
	}

	void toggleSampleShading()
	{
		useSampleShading = !useSampleShading;
		reBuildCommandBuffers();
	}

	virtual void keyPressed(uint32_t keyCode)
	{
		switch (keyCode)
		{
		case KEY_S:
		case GAMEPAD_BUTTON_A:
			toggleSampleShading();
			break;
		}
	}

    // Returns the maximum sample count usable by the platform
    vk::SampleCountFlagBits getMaxUsableSampleCount()
    {
        vk::SampleCountFlags counts = std::min(deviceProperties.limits.framebufferColorSampleCounts,
                                            deviceProperties.limits.framebufferDepthSampleCounts);

        if (counts & vk::SampleCountFlagBits::e64) { return vk::SampleCountFlagBits::e64; }
        if (counts & vk::SampleCountFlagBits::e32) { return vk::SampleCountFlagBits::e32; }
        if (counts & vk::SampleCountFlagBits::e16) { return vk::SampleCountFlagBits::e16; }
        if (counts & vk::SampleCountFlagBits::e8) { return vk::SampleCountFlagBits::e8; }
        if (counts & vk::SampleCountFlagBits::e4) { return vk::SampleCountFlagBits::e4; }
        if (counts & vk::SampleCountFlagBits::e2) { return vk::SampleCountFlagBits::e2; }
        return vk::SampleCountFlagBits::e1;
    }

};

VULKAN_EXAMPLE_MAIN()
