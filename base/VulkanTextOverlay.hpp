/*
* Text overlay class for displaying debug information
*
* Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#pragma once

#include <vulkan/vulkan.hpp>
#include "VulkanDebug.h"
#include "VulkanDevice.hpp"

#if defined(__ANDROID__)
#include "VulkanAndroid.h"
#endif

#include "../external/stb/stb_font_consolas_24_latin1.inl"

// Defines for the STB font used
// STB font files can be found at http://nothings.org/stb/font/
#define STB_FONT_NAME stb_font_consolas_24_latin1
#define STB_FONT_WIDTH STB_FONT_consolas_24_latin1_BITMAP_WIDTH
#define STB_FONT_HEIGHT STB_FONT_consolas_24_latin1_BITMAP_HEIGHT 
#define STB_FIRST_CHAR STB_FONT_consolas_24_latin1_FIRST_CHAR
#define STB_NUM_CHARS STB_FONT_consolas_24_latin1_NUM_CHARS

// Max. number of chars the text overlay buffer can hold
#define MAX_CHAR_COUNT 1024

/**
* @brief Mostly self-contained text overlay class
* @note Will only work with compatible render passes
*/ 
class VulkanTextOverlay
{
private:
	vks::VulkanDevice *vulkanDevice;

	vk::Queue queue;
	vk::Format colorFormat;
	vk::Format depthFormat;

	uint32_t *frameBufferWidth;
	uint32_t *frameBufferHeight;

	vk::Sampler sampler;
	vk::Image image;
	vk::ImageView view;
	vks::Buffer vertexBuffer;
	vk::DeviceMemory imageMemory;
	vk::DescriptorPool descriptorPool;
	vk::DescriptorSetLayout descriptorSetLayout;
	vk::DescriptorSet descriptorSet;
	vk::PipelineLayout pipelineLayout;
	vk::PipelineCache pipelineCache;
	vk::Pipeline pipeline;
	vk::RenderPass renderPass;
	vk::CommandPool commandPool;
	std::vector<vk::Framebuffer*> frameBuffers;
	std::vector<vk::PipelineShaderStageCreateInfo> shaderStages;
	vk::Fence fence;

	// Used during text updates
	glm::vec4 *mappedLocal = nullptr;

	stb_fontchar stbFontData[STB_NUM_CHARS];
	uint32_t numLetters;

public:

	enum TextAlign { alignLeft, alignCenter, alignRight };

	bool visible = true;
	bool invalidated = false;

	float scale = 1.0f;

	std::vector<vk::CommandBuffer> cmdBuffers;

	/**
	* Default constructor
	*
	* @param vulkanDevice Pointer to a valid VulkanDevice
	*/
	VulkanTextOverlay(
		vks::VulkanDevice *vulkanDevice,
		vk::Queue queue,
		std::vector<vk::Framebuffer> &framebuffers,
		vk::Format colorformat,
		vk::Format depthformat,
		uint32_t *framebufferwidth,
		uint32_t *framebufferheight,
		std::vector<vk::PipelineShaderStageCreateInfo> shaderstages)
	{
		this->vulkanDevice = vulkanDevice;
		this->queue = queue;
		this->colorFormat = colorformat;
		this->depthFormat = depthformat;

		this->frameBuffers.resize(framebuffers.size());
		for (uint32_t i = 0; i < framebuffers.size(); i++)
		{
			this->frameBuffers[i] = &framebuffers[i];
		}

		this->shaderStages = shaderstages;

		this->frameBufferWidth = framebufferwidth;
		this->frameBufferHeight = framebufferheight;

#if defined(__ANDROID__)		
		// Scale text on Android devices with high DPI
		if (vks::android::screenDensity >= ACONFIGURATION_DENSITY_XXHIGH) {
			LOGD("XXHIGH");
			scale = 2.0f;
		} 
		else if (vks::android::screenDensity >= ACONFIGURATION_DENSITY_XHIGH) {
			LOGD("XHIGH");
			scale = 1.5f;
		} 
		else if (vks::android::screenDensity >= ACONFIGURATION_DENSITY_HIGH) {
			LOGD("HIGH");
			scale = 1.25f;
		};
#endif

		cmdBuffers.resize(framebuffers.size());
		prepareResources();
		prepareRenderPass();
		preparePipeline();
	}

	/**
	* Default destructor, frees up all Vulkan resources acquired by the text overlay
	*/
	~VulkanTextOverlay()
	{
		// Free up all Vulkan resources requested by the text overlay
		vertexBuffer.destroy();
		vulkanDevice->logicalDevice.destroySampler(sampler);
		vulkanDevice->logicalDevice.destroyImage(image);
		vulkanDevice->logicalDevice.destroyImageView(view);
		vulkanDevice->logicalDevice.freeMemory(imageMemory);
		vulkanDevice->logicalDevice.destroyDescriptorSetLayout(descriptorSetLayout);
		vulkanDevice->logicalDevice.destroyDescriptorPool(descriptorPool);
		vulkanDevice->logicalDevice.destroyPipelineLayout(pipelineLayout);
		vulkanDevice->logicalDevice.destroyPipelineCache(pipelineCache);
		vulkanDevice->logicalDevice.destroyPipeline(pipeline);
		vulkanDevice->logicalDevice.destroyRenderPass(renderPass);
		vulkanDevice->logicalDevice.freeCommandBuffers(commandPool, cmdBuffers);
		vulkanDevice->logicalDevice.destroyCommandPool(commandPool);
		vulkanDevice->logicalDevice.destroyFence(fence);
	}

	/**
	* Prepare all vulkan resources required to render the font
	* The text overlay uses separate resources for descriptors (pool, sets, layouts), pipelines and command buffers
	*/
	void prepareResources()
	{
		static unsigned char font24pixels[STB_FONT_HEIGHT][STB_FONT_WIDTH];
		STB_FONT_NAME(stbFontData, font24pixels, STB_FONT_HEIGHT);

		// Command buffer

		// Pool
		vk::CommandPoolCreateInfo cmdPoolInfo = {};

		cmdPoolInfo.queueFamilyIndex = vulkanDevice->queueFamilyIndices.graphics; 
		cmdPoolInfo.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
		commandPool = vulkanDevice->logicalDevice.createCommandPool(cmdPoolInfo);

		vk::CommandBufferAllocateInfo cmdBufAllocateInfo =
			vks::initializers::commandBufferAllocateInfo(
				commandPool,
				vk::CommandBufferLevel::ePrimary,
				(uint32_t)cmdBuffers.size());

		cmdBuffers = vulkanDevice->logicalDevice.allocateCommandBuffers(cmdBufAllocateInfo);

		// Vertex buffer
		vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eVertexBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			&vertexBuffer,
			MAX_CHAR_COUNT * sizeof(glm::vec4));

		// Map persistent
		vertexBuffer.map();

		// Font texture
		vk::ImageCreateInfo imageInfo = vks::initializers::imageCreateInfo();
		imageInfo.imageType = vk::ImageType::e2D;
		imageInfo.format = vk::Format::eR8Unorm;
		imageInfo.extent = vk::Extent3D{ STB_FONT_WIDTH, STB_FONT_HEIGHT, 1 };
		imageInfo.mipLevels = 1;
		imageInfo.arrayLayers = 1;
		imageInfo.samples = vk::SampleCountFlagBits::e1;
		imageInfo.tiling = vk::ImageTiling::eOptimal;
		imageInfo.usage = vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled;
		imageInfo.sharingMode = vk::SharingMode::eExclusive;
		imageInfo.initialLayout = vk::ImageLayout::ePreinitialized;
		image = vulkanDevice->logicalDevice.createImage(imageInfo);

		vk::MemoryRequirements memReqs;
		vk::MemoryAllocateInfo allocInfo = vks::initializers::memoryAllocateInfo();
		memReqs = vulkanDevice->logicalDevice.getImageMemoryRequirements(image);
		allocInfo.allocationSize = memReqs.size;
		allocInfo.memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);
		imageMemory = vulkanDevice->logicalDevice.allocateMemory(allocInfo);
		vulkanDevice->logicalDevice.bindImageMemory(image, imageMemory, 0);

		// Staging
		vks::Buffer stagingBuffer;

		vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eTransferSrc,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			&stagingBuffer,
			allocInfo.allocationSize);

		stagingBuffer.map();
		memcpy(stagingBuffer.mapped, &font24pixels[0][0], STB_FONT_WIDTH * STB_FONT_HEIGHT);	// Only one channel, so data size = W * H (*R8)
		stagingBuffer.unmap();

		// Copy to image
		vk::CommandBuffer copyCmd;
		cmdBufAllocateInfo.commandBufferCount = 1;
		copyCmd = vulkanDevice->logicalDevice.allocateCommandBuffers(cmdBufAllocateInfo)[0];

		vk::CommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();
		copyCmd.begin(cmdBufInfo);

		// Prepare for transfer
		vks::tools::setImageLayout(
			copyCmd,
			image,
			vk::ImageAspectFlagBits::eColor,
			vk::ImageLayout::ePreinitialized,
			vk::ImageLayout::eTransferDstOptimal);

		vk::BufferImageCopy bufferCopyRegion = {};
		bufferCopyRegion.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
		bufferCopyRegion.imageSubresource.mipLevel = 0;
		bufferCopyRegion.imageSubresource.layerCount = 1;
		bufferCopyRegion.imageExtent.width = STB_FONT_WIDTH;
		bufferCopyRegion.imageExtent.height = STB_FONT_HEIGHT;
		bufferCopyRegion.imageExtent.depth = 1;

		copyCmd.copyBufferToImage(
			stagingBuffer.buffer,
			image,
			vk::ImageLayout::eTransferDstOptimal,
			bufferCopyRegion
			);

		// Prepare for shader read
		vks::tools::setImageLayout(
			copyCmd,
			image,
			vk::ImageAspectFlagBits::eColor,
			vk::ImageLayout::eTransferDstOptimal,
			vk::ImageLayout::eShaderReadOnlyOptimal);

		copyCmd.end();

		vk::SubmitInfo submitInfo = vks::initializers::submitInfo();
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &copyCmd;

		queue.submit(submitInfo, vk::Fence(nullptr));
		queue.waitIdle();

		stagingBuffer.destroy();

		vulkanDevice->logicalDevice.freeCommandBuffers(commandPool, copyCmd);

		vk::ImageViewCreateInfo imageViewInfo = vks::initializers::imageViewCreateInfo();
		imageViewInfo.image = image;
		imageViewInfo.viewType = vk::ImageViewType::e2D;
		imageViewInfo.format = imageInfo.format;
		imageViewInfo.components = { vk::ComponentSwizzle::eR, vk::ComponentSwizzle::eG, vk::ComponentSwizzle::eB,	vk::ComponentSwizzle::eA };
		imageViewInfo.subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 };
		view = vulkanDevice->logicalDevice.createImageView(imageViewInfo);

		// Sampler
		vk::SamplerCreateInfo samplerInfo = vks::initializers::samplerCreateInfo();
		samplerInfo.magFilter = vk::Filter::eLinear;
		samplerInfo.minFilter = vk::Filter::eLinear;
		samplerInfo.mipmapMode = vk::SamplerMipmapMode::eLinear;
		samplerInfo.addressModeU = vk::SamplerAddressMode::eRepeat;
		samplerInfo.addressModeV = vk::SamplerAddressMode::eRepeat;
		samplerInfo.addressModeW = vk::SamplerAddressMode::eRepeat;
		samplerInfo.mipLodBias = 0.0f;
		samplerInfo.compareOp = vk::CompareOp::eNever;
		samplerInfo.minLod = 0.0f;
		samplerInfo.maxLod = 1.0f;
		samplerInfo.borderColor = vk::BorderColor::eFloatOpaqueWhite;
		samplerInfo.maxAnisotropy = 1.0f;
		sampler = vulkanDevice->logicalDevice.createSampler(samplerInfo);

		// Descriptor
		// Font uses a separate descriptor pool
		std::array<vk::DescriptorPoolSize, 1> poolSizes;
		poolSizes[0] = vks::initializers::descriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, 1);

		vk::DescriptorPoolCreateInfo descriptorPoolInfo =
			vks::initializers::descriptorPoolCreateInfo(
				static_cast<uint32_t>(poolSizes.size()),
				poolSizes.data(),
				1);

		descriptorPool = vulkanDevice->logicalDevice.createDescriptorPool(descriptorPoolInfo);

		// Descriptor set layout
		std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings(1);
		setLayoutBindings[0] = vks::initializers::descriptorSetLayoutBinding(vk::DescriptorType::eCombinedImageSampler, vk::ShaderStageFlagBits::eFragment, 0);

		vk::DescriptorSetLayoutCreateInfo descriptorSetLayoutInfo =
			vks::initializers::descriptorSetLayoutCreateInfo(
				setLayoutBindings);

		descriptorSetLayout = vulkanDevice->logicalDevice.createDescriptorSetLayout(descriptorSetLayoutInfo);

		// Pipeline layout
		vk::PipelineLayoutCreateInfo pipelineLayoutInfo =
			vks::initializers::pipelineLayoutCreateInfo(
				&descriptorSetLayout,
				1);

		pipelineLayout = vulkanDevice->logicalDevice.createPipelineLayout(pipelineLayoutInfo);

		// Descriptor set
		vk::DescriptorSetAllocateInfo descriptorSetAllocInfo =
			vks::initializers::descriptorSetAllocateInfo(
				descriptorPool,
				&descriptorSetLayout,
				1);

		descriptorSet = vulkanDevice->logicalDevice.allocateDescriptorSets(descriptorSetAllocInfo)[0];

		vk::DescriptorImageInfo texDescriptor =
			vks::initializers::descriptorImageInfo(
				sampler,
				view,
				vk::ImageLayout::eGeneral);

		std::vector<vk::WriteDescriptorSet> writeDescriptorSets;
		writeDescriptorSets.push_back(vks::initializers::writeDescriptorSet(descriptorSet, vk::DescriptorType::eCombinedImageSampler, 0, &texDescriptor));
		vulkanDevice->logicalDevice.updateDescriptorSets(writeDescriptorSets, nullptr);

		// Pipeline cache
		vk::PipelineCacheCreateInfo pipelineCacheCreateInfo = {};

		pipelineCache = vulkanDevice->logicalDevice.createPipelineCache(pipelineCacheCreateInfo);

		// Command buffer execution fence
		vk::FenceCreateInfo fenceCreateInfo = vks::initializers::fenceCreateInfo();
		fence = vulkanDevice->logicalDevice.createFence(fenceCreateInfo);
	}

	/**
	* Prepare a separate pipeline for the font rendering decoupled from the main application
	*/
	void preparePipeline()
	{
		vk::PipelineInputAssemblyStateCreateInfo inputAssemblyState =
			vks::initializers::pipelineInputAssemblyStateCreateInfo(
				vk::PrimitiveTopology::eTriangleStrip,
				vk::PipelineInputAssemblyStateCreateFlags(),
				VK_FALSE);

		vk::PipelineRasterizationStateCreateInfo rasterizationState =
			vks::initializers::pipelineRasterizationStateCreateInfo(
				vk::PolygonMode::eFill,
				vk::CullModeFlagBits::eBack,
				vk::FrontFace::eClockwise);

		// Enable blending
		vk::PipelineColorBlendAttachmentState blendAttachmentState =
			vks::initializers::pipelineColorBlendAttachmentState(
				vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA, 
				VK_TRUE);

		blendAttachmentState.srcColorBlendFactor = vk::BlendFactor::eOne;
		blendAttachmentState.dstColorBlendFactor = vk::BlendFactor::eOne;
		blendAttachmentState.colorBlendOp = vk::BlendOp::eAdd;
		blendAttachmentState.srcAlphaBlendFactor = vk::BlendFactor::eOne;
		blendAttachmentState.dstAlphaBlendFactor = vk::BlendFactor::eOne;
		blendAttachmentState.alphaBlendOp = vk::BlendOp::eAdd;
		blendAttachmentState.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;

		vk::PipelineColorBlendStateCreateInfo colorBlendState =
			vks::initializers::pipelineColorBlendStateCreateInfo(
				1,
				&blendAttachmentState);

		vk::PipelineDepthStencilStateCreateInfo depthStencilState =
			vks::initializers::pipelineDepthStencilStateCreateInfo(
				VK_FALSE,
				VK_FALSE,
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

		std::array<vk::VertexInputBindingDescription, 2> vertexBindings = {};
		vertexBindings[0] = vks::initializers::vertexInputBindingDescription(0, sizeof(glm::vec4), vk::VertexInputRate::eVertex);
		vertexBindings[1] = vks::initializers::vertexInputBindingDescription(1, sizeof(glm::vec4), vk::VertexInputRate::eVertex);

		std::array<vk::VertexInputAttributeDescription, 2> vertexAttribs = {};
		// Position
		vertexAttribs[0] = vks::initializers::vertexInputAttributeDescription(0, 0, vk::Format::eR32G32Sfloat, 0);
		// UV
		vertexAttribs[1] = vks::initializers::vertexInputAttributeDescription(1, 1, vk::Format::eR32G32Sfloat, sizeof(glm::vec2));

		vk::PipelineVertexInputStateCreateInfo inputState = vks::initializers::pipelineVertexInputStateCreateInfo();
		inputState.vertexBindingDescriptionCount = static_cast<uint32_t>(vertexBindings.size());
		inputState.pVertexBindingDescriptions = vertexBindings.data();
		inputState.vertexAttributeDescriptionCount = static_cast<uint32_t>(vertexAttribs.size());
		inputState.pVertexAttributeDescriptions = vertexAttribs.data();

		vk::GraphicsPipelineCreateInfo pipelineCreateInfo =
			vks::initializers::pipelineCreateInfo(
				pipelineLayout,
				renderPass);

		pipelineCreateInfo.pVertexInputState = &inputState;
		pipelineCreateInfo.pInputAssemblyState = &inputAssemblyState;
		pipelineCreateInfo.pRasterizationState = &rasterizationState;
		pipelineCreateInfo.pColorBlendState = &colorBlendState;
		pipelineCreateInfo.pMultisampleState = &multisampleState;
		pipelineCreateInfo.pViewportState = &viewportState;
		pipelineCreateInfo.pDepthStencilState = &depthStencilState;
		pipelineCreateInfo.pDynamicState = &dynamicState;
		pipelineCreateInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
		pipelineCreateInfo.pStages = shaderStages.data();

		pipeline = vulkanDevice->logicalDevice.createGraphicsPipelines(pipelineCache, pipelineCreateInfo)[0];
	}

	/**
	* Prepare a separate render pass for rendering the text as an overlay
	*/
	void prepareRenderPass()
	{
		vk::AttachmentDescription attachments[2] = {};

		// Color attachment
		attachments[0].format = colorFormat;
		attachments[0].samples = vk::SampleCountFlagBits::e1;
		// Don't clear the framebuffer (like the renderpass from the example does)
		attachments[0].loadOp = vk::AttachmentLoadOp::eLoad;
		attachments[0].storeOp = vk::AttachmentStoreOp::eStore;
		attachments[0].stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
		attachments[0].stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
		attachments[0].initialLayout = vk::ImageLayout::eUndefined;
		attachments[0].finalLayout = vk::ImageLayout::ePresentSrcKHR;

		// Depth attachment
		attachments[1].format = depthFormat;
		attachments[1].samples = vk::SampleCountFlagBits::e1;
		attachments[1].loadOp = vk::AttachmentLoadOp::eDontCare;
		attachments[1].storeOp = vk::AttachmentStoreOp::eDontCare;
		attachments[1].stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
		attachments[1].stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
		attachments[1].initialLayout = vk::ImageLayout::eUndefined;
		attachments[1].finalLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal;

		vk::AttachmentReference colorReference = {};
		colorReference.attachment = 0;
		colorReference.layout = vk::ImageLayout::eColorAttachmentOptimal;

		vk::AttachmentReference depthReference = {};
		depthReference.attachment = 1;
		depthReference.layout = vk::ImageLayout::eDepthStencilAttachmentOptimal;

		vk::SubpassDependency subpassDependencies[2] = {};

		// Transition from final to initial (VK_SUBPASS_EXTERNAL refers to all commmands executed outside of the actual renderpass)
		subpassDependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
		subpassDependencies[0].dstSubpass = 0;
		subpassDependencies[0].srcStageMask = vk::PipelineStageFlagBits::eBottomOfPipe;
		subpassDependencies[0].dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
		subpassDependencies[0].srcAccessMask = vk::AccessFlagBits::eMemoryRead;
		subpassDependencies[0].dstAccessMask = vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite;
		subpassDependencies[0].dependencyFlags = vk::DependencyFlagBits::eByRegion;

		// Transition from initial to final
		subpassDependencies[1].srcSubpass = 0;
		subpassDependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
		subpassDependencies[1].srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
		subpassDependencies[1].dstStageMask = vk::PipelineStageFlagBits::eBottomOfPipe;
		subpassDependencies[1].srcAccessMask = vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite;
		subpassDependencies[1].dstAccessMask = vk::AccessFlagBits::eMemoryRead;
		subpassDependencies[1].dependencyFlags = vk::DependencyFlagBits::eByRegion;

		vk::SubpassDescription subpassDescription = {};
		subpassDescription.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
		subpassDescription.inputAttachmentCount = 0;
		subpassDescription.pInputAttachments = NULL;
		subpassDescription.colorAttachmentCount = 1;
		subpassDescription.pColorAttachments = &colorReference;
		subpassDescription.pResolveAttachments = NULL;
		subpassDescription.pDepthStencilAttachment = &depthReference;
		subpassDescription.preserveAttachmentCount = 0;
		subpassDescription.pPreserveAttachments = NULL;

		vk::RenderPassCreateInfo renderPassInfo = {};
		renderPassInfo.attachmentCount = 2;
		renderPassInfo.pAttachments = attachments;
		renderPassInfo.subpassCount = 1;
		renderPassInfo.pSubpasses = &subpassDescription;
		renderPassInfo.dependencyCount = 2;
		renderPassInfo.pDependencies = subpassDependencies;

		renderPass = vulkanDevice->logicalDevice.createRenderPass(renderPassInfo);
	}

	/**
	* Maps the buffer, resets letter count
	*/
	void beginTextUpdate()
	{
		mappedLocal = (glm::vec4*)vertexBuffer.mapped;
		numLetters = 0;
	}

	/**
	* Add text to the current buffer
	*
	* @param text Text to add
	* @param x x position of the text to add in window coordinate space
	* @param y y position of the text to add in window coordinate space
	* @param align Alignment for the new text (left, right, center)
	*/
	void addText(std::string text, float x, float y, TextAlign align)
	{
		assert(vertexBuffer.mapped != nullptr);

		if (align == alignLeft) {
			x *= scale;
		};

		y *= scale;

		const float charW = (1.5f * scale) / *frameBufferWidth;
		const float charH = (1.5f * scale) / *frameBufferHeight;

		float fbW = (float)*frameBufferWidth;
		float fbH = (float)*frameBufferHeight;
		x = (x / fbW * 2.0f) - 1.0f;
		y = (y / fbH * 2.0f) - 1.0f;

		// Calculate text width
		float textWidth = 0;
		for (auto letter : text)
		{
			stb_fontchar *charData = &stbFontData[(uint32_t)letter - STB_FIRST_CHAR];
			textWidth += charData->advance * charW;
		}

		switch (align)
		{
		case alignRight:
			x -= textWidth;
			break;
		case alignCenter:
			x -= textWidth / 2.0f;
			break;
		case alignLeft:
			break;
		}

		// Generate a uv mapped quad per char in the new text
		for (auto letter : text)
		{
			stb_fontchar *charData = &stbFontData[(uint32_t)letter - STB_FIRST_CHAR];

			mappedLocal->x = (x + (float)charData->x0 * charW);
			mappedLocal->y = (y + (float)charData->y0 * charH);
			mappedLocal->z = charData->s0;
			mappedLocal->w = charData->t0;
			mappedLocal++;

			mappedLocal->x = (x + (float)charData->x1 * charW);
			mappedLocal->y = (y + (float)charData->y0 * charH);
			mappedLocal->z = charData->s1;
			mappedLocal->w = charData->t0;
			mappedLocal++;

			mappedLocal->x = (x + (float)charData->x0 * charW);
			mappedLocal->y = (y + (float)charData->y1 * charH);
			mappedLocal->z = charData->s0;
			mappedLocal->w = charData->t1;
			mappedLocal++;

			mappedLocal->x = (x + (float)charData->x1 * charW);
			mappedLocal->y = (y + (float)charData->y1 * charH);
			mappedLocal->z = charData->s1;
			mappedLocal->w = charData->t1;
			mappedLocal++;

			x += charData->advance * charW;

			numLetters++;
		}
	}

	/**
	* Unmap buffer and update command buffers
	*/
	void endTextUpdate()
	{
		updateCommandBuffers();
	}

	/**
	* Update the command buffers to reflect text changes
	*/
	void updateCommandBuffers()
	{
		vk::CommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();

		vk::RenderPassBeginInfo renderPassBeginInfo = vks::initializers::renderPassBeginInfo();
		renderPassBeginInfo.renderPass = renderPass;
		renderPassBeginInfo.renderArea.extent.width = *frameBufferWidth;
		renderPassBeginInfo.renderArea.extent.height = *frameBufferHeight;
		// None of the attachments will be cleared
		renderPassBeginInfo.clearValueCount = 0;
		renderPassBeginInfo.pClearValues = nullptr;

		for (size_t i = 0; i < cmdBuffers.size(); ++i)
		{
			renderPassBeginInfo.framebuffer = *frameBuffers[i];

			cmdBuffers[i].begin(cmdBufInfo);

			if (vks::debugmarker::active)
			{
				vks::debugmarker::beginRegion(cmdBuffers[i], "Text overlay", glm::vec4(1.0f, 0.94f, 0.3f, 1.0f));
			}

			cmdBuffers[i].beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);

			vk::Viewport viewport = vks::initializers::viewport((float)*frameBufferWidth, (float)*frameBufferHeight, 0.0f, 1.0f);
			cmdBuffers[i].setViewport(0, viewport);

			vk::Rect2D scissor = vks::initializers::rect2D(*frameBufferWidth, *frameBufferHeight, 0, 0);
			cmdBuffers[i].setScissor(0, scissor);
			
			cmdBuffers[i].bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);
			cmdBuffers[i].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSet, nullptr);

			vk::DeviceSize offsets = 0;
			cmdBuffers[i].bindVertexBuffers(0, vertexBuffer.buffer, offsets);
			cmdBuffers[i].bindVertexBuffers(1, vertexBuffer.buffer, offsets);
			for (uint32_t j = 0; j < numLetters; j++)
			{
				vkCmdDraw(cmdBuffers[i], 4, 1, j * 4, 0);
			}

			cmdBuffers[i].endRenderPass();

			if (vks::debugmarker::active)
			{
				vks::debugmarker::endRegion(cmdBuffers[i]);
			}

			cmdBuffers[i].end();
		}
	}

	/**
	* Submit the text command buffers to a queue
	*/
	void submit(vk::Queue queue, uint32_t bufferindex, vk::SubmitInfo submitInfo)
	{
		if (!visible)
		{
			return;
		}

		submitInfo.pCommandBuffers = &cmdBuffers[bufferindex];
		submitInfo.commandBufferCount = 1;

		queue.submit(submitInfo, fence);

		vulkanDevice->logicalDevice.waitForFences(fence, VK_TRUE, UINT64_MAX);
		vulkanDevice->logicalDevice.resetFences(fence);
	}

	/**
	* Reallocate command buffers for the text overlay
	* @note Frees the existing command buffers
	*/
	void reallocateCommandBuffers()
	{
		vulkanDevice->logicalDevice.freeCommandBuffers(commandPool, cmdBuffers);

		vk::CommandBufferAllocateInfo cmdBufAllocateInfo =
			vks::initializers::commandBufferAllocateInfo(
				commandPool,
				vk::CommandBufferLevel::ePrimary,
				static_cast<uint32_t>(cmdBuffers.size()));

		cmdBuffers = vulkanDevice->logicalDevice.allocateCommandBuffers(cmdBufAllocateInfo);
	}

};