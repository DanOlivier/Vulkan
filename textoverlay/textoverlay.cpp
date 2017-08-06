/*
* Vulkan Example - Text overlay rendering on-top of an existing scene using a separate render pass
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
#include <sstream>
#include <iomanip>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/matrix_inverse.hpp>

#include <vulkan/vulkan.hpp>

#include "vulkanexamplebase.h"
#include "VulkanDevice.hpp"
#include "VulkanBuffer.hpp"
#include "VulkanModel.hpp"
#include "VulkanTexture.hpp"

#include "../external/stb/stb_font_consolas_24_latin1.inl"

#define VERTEX_BUFFER_BIND_ID 0
#define ENABLE_VALIDATION false

// Defines for the STB font used
// STB font files can be found at http://nothings.org/stb/font/
#define STB_FONT_NAME stb_font_consolas_24_latin1
#define STB_FONT_WIDTH STB_FONT_consolas_24_latin1_BITMAP_WIDTH
#define STB_FONT_HEIGHT STB_FONT_consolas_24_latin1_BITMAP_HEIGHT 
#define STB_FIRST_CHAR STB_FONT_consolas_24_latin1_FIRST_CHAR
#define STB_NUM_CHARS STB_FONT_consolas_24_latin1_NUM_CHARS

// Max. number of chars the text overlay buffer can hold
#define TEXTOVERLAY_MAX_CHAR_COUNT 2048

// Mostly self-contained text overlay class
class TextOverlay
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
	vk::Buffer buffer;
	vk::DeviceMemory memory;
	vk::DeviceMemory imageMemory;
	vk::DescriptorPool descriptorPool;
	vk::DescriptorSetLayout descriptorSetLayout;
	vk::DescriptorSet descriptorSet;
	vk::PipelineLayout pipelineLayout;
	vk::PipelineCache pipelineCache;
	vk::Pipeline pipeline;
	vk::RenderPass renderPass;
	vk::CommandPool commandPool;
	std::vector<vk::CommandBuffer> cmdBuffers;
	std::vector<vk::Framebuffer*> frameBuffers;
	std::vector<vk::PipelineShaderStageCreateInfo> shaderStages;

	// Pointer to mapped vertex buffer
	glm::vec4 *mapped = nullptr;

	stb_fontchar stbFontData[STB_NUM_CHARS];
	uint32_t numLetters;
public:

	enum TextAlign { alignLeft, alignCenter, alignRight };

	bool visible = true;

	TextOverlay(
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

		cmdBuffers.resize(framebuffers.size());
		prepareResources();
		prepareRenderPass();
		preparePipeline();
	}

	~TextOverlay()
	{
		// Free up all Vulkan resources requested by the text overlay
		vulkanDevice->logicalDevice.destroySampler(sampler);
		vulkanDevice->logicalDevice.destroyImage(image);
		vulkanDevice->logicalDevice.destroyImageView(view);
		vulkanDevice->logicalDevice.destroyBuffer(buffer);
		vulkanDevice->logicalDevice.freeMemory(memory);
		vulkanDevice->logicalDevice.freeMemory(imageMemory);
		vulkanDevice->logicalDevice.destroyDescriptorSetLayout(descriptorSetLayout);
		vulkanDevice->logicalDevice.destroyDescriptorPool(descriptorPool);
		vulkanDevice->logicalDevice.destroyPipelineLayout(pipelineLayout);
		vulkanDevice->logicalDevice.destroyPipelineCache(pipelineCache);
		vulkanDevice->logicalDevice.destroyPipeline(pipeline);
		vulkanDevice->logicalDevice.destroyRenderPass(renderPass);
		vulkanDevice->logicalDevice.destroyCommandPool(commandPool);
	}

	// Prepare all vulkan resources required to render the font
	// The text overlay uses separate resources for descriptors (pool, sets, layouts), pipelines and command buffers
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

		cmdBuffers = vulkanDevice->logicalDevice.allocateCommandBuffers(cmdBufAllocateInfo)[0];

		// Vertex buffer
		vk::DeviceSize bufferSize = TEXTOVERLAY_MAX_CHAR_COUNT * sizeof(glm::vec4);

		vk::BufferCreateInfo bufferInfo = vks::initializers::bufferCreateInfo(vk::BufferUsageFlagBits::eVertexBuffer, bufferSize);
		buffer = vulkanDevice->logicalDevice.createBuffer(bufferInfo);

		vk::MemoryRequirements memReqs;
		vk::MemoryAllocateInfo allocInfo = vks::initializers::memoryAllocateInfo();

		memReqs = vulkanDevice->logicalDevice.getBufferMemoryRequirements(buffer);
		allocInfo.allocationSize = memReqs.size;
		allocInfo.memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

		memory = vulkanDevice->logicalDevice.allocateMemory(allocInfo);
		vulkanDevice->logicalDevice.bindBufferMemory(buffer, memory, 0);

		// Font texture
		vk::ImageCreateInfo imageInfo = vks::initializers::imageCreateInfo();
		imageInfo.imageType = vk::ImageType::e2D;
		imageInfo.format = vk::Format::eR8Unorm;
		imageInfo.extent.width = STB_FONT_WIDTH;
		imageInfo.extent.height = STB_FONT_HEIGHT;
		imageInfo.extent.depth = 1;
		imageInfo.mipLevels = 1;
		imageInfo.arrayLayers = 1;
		imageInfo.samples = vk::SampleCountFlagBits::e1;
		imageInfo.tiling = vk::ImageTiling::eOptimal;
		imageInfo.usage = vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled;
		imageInfo.sharingMode = vk::SharingMode::eExclusive;
		imageInfo.initialLayout = vk::ImageLayout::eUndefined;

		image = vulkanDevice->logicalDevice.createImage(imageInfo);

		memReqs = vulkanDevice->logicalDevice.getImageMemoryRequirements(image);
		allocInfo.allocationSize = memReqs.size;
		allocInfo.memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);

		imageMemory = vulkanDevice->logicalDevice.allocateMemory(allocInfo);
		vulkanDevice->logicalDevice.bindImageMemory(image, imageMemory, 0);

		// Staging

		struct {
			vk::DeviceMemory memory;
			vk::Buffer buffer;
		} stagingBuffer;

		vk::BufferCreateInfo bufferCreateInfo = vks::initializers::bufferCreateInfo();
		bufferCreateInfo.size = allocInfo.allocationSize;
		bufferCreateInfo.usage = vk::BufferUsageFlagBits::eTransferSrc;
		bufferCreateInfo.sharingMode = vk::SharingMode::eExclusive;

		stagingBuffer.buffer = vulkanDevice->logicalDevice.createBuffer(bufferCreateInfo);

		// Get memory requirements for the staging buffer (alignment, memory type bits)
		memReqs = vulkanDevice->logicalDevice.getBufferMemoryRequirements(stagingBuffer.buffer);

		allocInfo.allocationSize = memReqs.size;
		// Get memory type index for a host visible buffer
		allocInfo.memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

		stagingBuffer.memory = vulkanDevice->logicalDevice.allocateMemory(allocInfo);
		vulkanDevice->logicalDevice.bindBufferMemory(stagingBuffer.buffer, stagingBuffer.memory, 0);

		uint8_t *data = (uint8_t*)vulkanDevice->logicalDevice.mapMemory(stagingBuffer.memory, 0, allocInfo.allocationSize, vk::MemoryMapFlags());
		// Size of the font texture is WIDTH * HEIGHT * 1 byte (only one channel)
		memcpy(data, &font24pixels[0][0], STB_FONT_WIDTH * STB_FONT_HEIGHT);
		vulkanDevice->logicalDevice.unmapMemory(stagingBuffer.memory);

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
			vk::ImageLayout::eUndefined,
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

		vulkanDevice->logicalDevice.freeCommandBuffers(commandPool, copyCmd);
		vulkanDevice->logicalDevice.freeMemory(stagingBuffer.memory);
		vulkanDevice->logicalDevice.destroyBuffer(stagingBuffer.buffer);

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
		std::array<vk::DescriptorSetLayoutBinding, 1> setLayoutBindings;
		setLayoutBindings[0] = vks::initializers::descriptorSetLayoutBinding(vk::DescriptorType::eCombinedImageSampler, vk::ShaderStageFlagBits::eFragment, 0);

		vk::DescriptorSetLayoutCreateInfo descriptorSetLayoutInfo =
			vks::initializers::descriptorSetLayoutCreateInfo(
				setLayoutBindings.data(),
				static_cast<uint32_t>(setLayoutBindings.size()));
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

		std::array<vk::WriteDescriptorSet, 1> writeDescriptorSets;
		writeDescriptorSets[0] = vks::initializers::writeDescriptorSet(descriptorSet, vk::DescriptorType::eCombinedImageSampler, 0, &texDescriptor);
		vulkanDevice->logicalDevice.updateDescriptorSets(writeDescriptorSets, nullptr);

		// Pipeline cache
		vk::PipelineCacheCreateInfo pipelineCacheCreateInfo = {};

		pipelineCache = vulkanDevice->logicalDevice.createPipelineCache(pipelineCacheCreateInfo);
	}

	// Prepare a separate pipeline for the font rendering decoupled from the main application
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

	// Prepare a separate render pass for rendering the text as an overlay
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
		attachments[1].loadOp = vk::AttachmentLoadOp::eClear;
		attachments[1].storeOp = vk::AttachmentStoreOp::eStore;
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

		// Use subpass dependencies for image layout transitions
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
		subpassDescription.flags = 0;
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

	// Map buffer 
	void beginTextUpdate()
	{
		mapped = vulkanDevice->logicalDevice.mapMemory(memory, 0, VK_WHOLE_SIZE, vk::MemoryMapFlags());
		numLetters = 0;
	}

	// Add text to the current buffer
	// todo : drop shadow? color attribute?
	void addText(std::string text, float x, float y, TextAlign align)
	{
		assert(mapped != nullptr);

		const float charW = 1.5f / *frameBufferWidth;
		const float charH = 1.5f / *frameBufferHeight;

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
		default:
			break;
		}

		// Generate a uv mapped quad per char in the new text
		for (auto letter : text)
		{
			stb_fontchar *charData = &stbFontData[(uint32_t)letter - STB_FIRST_CHAR];

			mapped->x = (x + (float)charData->x0 * charW);
			mapped->y = (y + (float)charData->y0 * charH);
			mapped->z = charData->s0;
			mapped->w = charData->t0;
			mapped++;

			mapped->x = (x + (float)charData->x1 * charW);
			mapped->y = (y + (float)charData->y0 * charH);
			mapped->z = charData->s1;
			mapped->w = charData->t0;
			mapped++;

			mapped->x = (x + (float)charData->x0 * charW);
			mapped->y = (y + (float)charData->y1 * charH);
			mapped->z = charData->s0;
			mapped->w = charData->t1;
			mapped++;

			mapped->x = (x + (float)charData->x1 * charW);
			mapped->y = (y + (float)charData->y1 * charH);
			mapped->z = charData->s1;
			mapped->w = charData->t1;
			mapped++;

			x += charData->advance * charW;

			numLetters++;
		}
	}

	// Unmap buffer and update command buffers
	void endTextUpdate()
	{
		vulkanDevice->logicalDevice.unmapMemory(memory);
		mapped = nullptr;
		updateCommandBuffers();
	}

	// Needs to be called by the application
	void updateCommandBuffers()
	{
		vk::CommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();

		vk::ClearValue clearValues[2];
		clearValues[1].color = { { 0.0f, 0.0f, 0.0f, 0.0f } };

		vk::RenderPassBeginInfo renderPassBeginInfo = vks::initializers::renderPassBeginInfo();
		renderPassBeginInfo.renderPass = renderPass;
		renderPassBeginInfo.renderArea.extent.width = *frameBufferWidth;
		renderPassBeginInfo.renderArea.extent.height = *frameBufferHeight;
		renderPassBeginInfo.clearValueCount = 2;
		renderPassBeginInfo.pClearValues = clearValues;

		for (uint32_t i = 0; i < cmdBuffers.size(); ++i)
		{
			renderPassBeginInfo.framebuffer = *frameBuffers[i];

			cmdBuffers[i].begin(cmdBufInfo);

			cmdBuffers[i].beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);

			vk::Viewport viewport = vks::initializers::viewport((float)*frameBufferWidth, (float)*frameBufferHeight, 0.0f, 1.0f);
			cmdBuffers[i].setViewport(0, viewport);

			vk::Rect2D scissor = vks::initializers::rect2D(*frameBufferWidth, *frameBufferHeight, 0, 0);
			cmdBuffers[i].setScissor(0, scissor);

			cmdBuffers[i].bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);
			cmdBuffers[i].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSet, nullptr);

			vk::DeviceSize offsets = 0;
			cmdBuffers[i].bindVertexBuffers(0, buffer, offsets);
			cmdBuffers[i].bindVertexBuffers(1, buffer, offsets);
			for (uint32_t j = 0; j < numLetters; j++)
			{
				vkCmdDraw(cmdBuffers[i], 4, 1, j * 4, 0);
			}


			cmdBuffers[i].endRenderPass();

			cmdBuffers[i].end();
		}
	}

	// Submit the text command buffers to a queue
	// Does a queue wait idle
	void submit(vk::Queue queue, uint32_t bufferindex)
	{
		if (!visible)
		{
			return;
		}

		vk::SubmitInfo submitInfo = {};

		submitInfo.pCommandBuffers = &cmdBuffers[bufferindex];

		queue.submit(submitInfo, vk::Fence(nullptr));
		queue.waitIdle();
	}

};

class VulkanExample : public VulkanExampleBase
{
public:
	TextOverlay *textOverlay = nullptr;

	// Vertex layout for the models
	vks::VertexLayout vertexLayout = vks::VertexLayout({
		vks::VERTEX_COMPONENT_POSITION,
		vks::VERTEX_COMPONENT_NORMAL,
		vks::VERTEX_COMPONENT_UV,
		vks::VERTEX_COMPONENT_COLOR,
	});

	struct {
		vks::Texture2D background;
		vks::Texture2D cube;
	} textures;

	struct {
		vks::Model cube;
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
		glm::vec4 lightPos = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
	} uboVS;

	struct {
		vk::Pipeline solid;
		vk::Pipeline background;
	} pipelines;

	vk::PipelineLayout pipelineLayout;
	vk::DescriptorSetLayout descriptorSetLayout;

	struct {
		vk::DescriptorSet background;
		vk::DescriptorSet cube;
	} descriptorSets;


	VulkanExample() : VulkanExampleBase(ENABLE_VALIDATION)
	{
		zoom = -4.5f;
		zoomSpeed = 2.5f;
		rotation = { -25.0f, 0.0f, 0.0f };
		title = "Vulkan Example - Text overlay";
		// Disable text overlay of the example base class
		enableTextOverlay = false;
	}

	~VulkanExample()
	{
		device.destroyPipeline(pipelines.solid);
		device.destroyPipeline(pipelines.background);
		device.destroyPipelineLayout(pipelineLayout);
		device.destroyDescriptorSetLayout(descriptorSetLayout);
		models.cube.destroy();
		textures.background.destroy();
		textures.cube.destroy();
		uniformBuffer.destroy();
		delete(textOverlay);
	}

	void buildCommandBuffers()
	{
		vk::CommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();

		vk::ClearValue clearValues[3];

		clearValues[0].color = vk::ClearColorValue{ std::array<float, 4>{ 0.0f, 0.0f, 0.0f, 1.0f } };
		clearValues[1].depthStencil = vk::ClearDepthStencilValue{ 1.0f, 0 };

		vk::RenderPassBeginInfo renderPassBeginInfo = vks::initializers::renderPassBeginInfo();
		renderPassBeginInfo.renderPass = renderPass;
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

			drawCmdBuffers[i].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSets.background, nullptr);

			std::vector<vk::DeviceSize> offsets = { 0 };
			drawCmdBuffers[i].bindVertexBuffers(VERTEX_BUFFER_BIND_ID, models.cube.vertices.buffer, offsets);
			drawCmdBuffers[i].bindIndexBuffer(models.cube.indices.buffer, 0, vk::IndexType::eUint32);

			// Background
			drawCmdBuffers[i].bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.background);
			// Vertices are generated by the vertex shader
			vkCmdDraw(drawCmdBuffers[i], 4, 1, 0, 0);

			// Cube
			drawCmdBuffers[i].bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.solid);
			drawCmdBuffers[i].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSets.cube, nullptr);
			drawCmdBuffers[i].drawIndexed(models.cube.indexCount, 1, 0, 0, 0);

			drawCmdBuffers[i].endRenderPass();

			drawCmdBuffers[i].end();
		}

		queue.waitIdle();
	}

	// Update the text buffer displayed by the text overlay
	void updateTextOverlay(void)
	{
		textOverlay->beginTextUpdate();

		textOverlay->addText(title, 5.0f, 5.0f, TextOverlay::alignLeft);

		std::stringstream ss;
		ss << std::fixed << std::setprecision(2) << (frameTimer * 1000.0f) << "ms (" << lastFPS << " fps)";
		textOverlay->addText(ss.str(), 5.0f, 25.0f, TextOverlay::alignLeft);

		textOverlay->addText(deviceProperties.deviceName, 5.0f, 45.0f, TextOverlay::alignLeft);
		
		// Display projected cube vertices
		for (int32_t x = -1; x <= 1; x += 2)
		{
			for (int32_t y = -1; y <= 1; y += 2)
			{
				for (int32_t z = -1; z <= 1; z += 2)
				{
					std::stringstream vpos;
					vpos << std::showpos << x << "/" << y << "/" << z;
					glm::vec3 projected = glm::project(glm::vec3((float)x, (float)y, (float)z), uboVS.model, uboVS.projection, glm::vec4(0, 0, (float)width, (float)height));
					textOverlay->addText(vpos.str(), projected.x, projected.y + (y > -1 ? 5.0f : -20.0f), TextOverlay::alignCenter);
				}
			}
		}

		// Display current model view matrix
		textOverlay->addText("model view matrix", (float)width, 5.0f, TextOverlay::alignRight);

		for (uint32_t i = 0; i < 4; i++)
		{
			ss.str("");
			ss << std::fixed << std::setprecision(2) << std::showpos;
			ss << uboVS.model[0][i] << " " << uboVS.model[1][i] << " " << uboVS.model[2][i] << " " << uboVS.model[3][i];
			textOverlay->addText(ss.str(), (float)width, 25.0f + (float)i * 20.0f, TextOverlay::alignRight);
		}

		glm::vec3 projected = glm::project(glm::vec3(0.0f), uboVS.model, uboVS.projection, glm::vec4(0, 0, (float)width, (float)height));
		textOverlay->addText("Uniform cube", projected.x, projected.y, TextOverlay::alignCenter);

#if defined(__ANDROID__)
		// toto
#else
		textOverlay->addText("Press \"space\" to toggle text overlay", 5.0f, 65.0f, TextOverlay::alignLeft);
		textOverlay->addText("Hold middle mouse button and drag to move", 5.0f, 85.0f, TextOverlay::alignLeft);
#endif
		textOverlay->endTextUpdate();
	}

	void loadAssets()
	{
		models.cube.loadFromFile(getAssetPath() + "models/cube.dae", vertexLayout, 1.0f, vulkanDevice, queue);

		// Textures
		std::string texFormatSuffix;
		vk::Format texFormat;
		// Get supported compressed texture format
		if (vulkanDevice->features.textureCompressionBC) {
			texFormatSuffix = "_bc3_unorm";
			texFormat = vk::Format::eBc3UnormBlock;
		}
		else if (vulkanDevice->features.textureCompressionASTC_LDR) {
			texFormatSuffix = "_astc_8x8_unorm";
			texFormat = vk::Format::eAstc8x8UnormBlock;
		}
		else if (vulkanDevice->features.textureCompressionETC2) {
			texFormatSuffix = "_etc2_unorm";
			texFormat = vk::Format::eEtc2R8G8B8A8UnormBlock;
		}
		else {
			vks::tools::exitFatal("Device does not support any compressed texture format!", "Error");
		}

		textures.background.loadFromFile(getAssetPath() + "textures/skysphere" + texFormatSuffix + ".ktx", texFormat, vulkanDevice, queue);
		textures.cube.loadFromFile(getAssetPath() + "textures/round_window" + texFormatSuffix + ".ktx", texFormat, vulkanDevice, queue);
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
		vertices.inputState.vertexBindingDescriptionCount = static_cast<uint32_t>(vertices.bindingDescriptions.size());
		vertices.inputState.pVertexBindingDescriptions = vertices.bindingDescriptions.data();
		vertices.inputState.vertexAttributeDescriptionCount = static_cast<uint32_t>(vertices.attributeDescriptions.size());
		vertices.inputState.pVertexAttributeDescriptions = vertices.attributeDescriptions.data();
	}

	void setupDescriptorPool()
	{
		std::vector<vk::DescriptorPoolSize> poolSizes =
		{
			vks::initializers::descriptorPoolSize(vk::DescriptorType::eUniformBuffer, 2),
			vks::initializers::descriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, 2),
		};

		vk::DescriptorPoolCreateInfo descriptorPoolInfo =
			vks::initializers::descriptorPoolCreateInfo(
				static_cast<uint32_t>(poolSizes.size()),
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
				static_cast<uint32_t>(setLayoutBindings.size()));

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

		// Background
		descriptorSets.background = device.allocateDescriptorSets(allocInfo)[0];

		vk::DescriptorImageInfo texDescriptor =
			vks::initializers::descriptorImageInfo(
				textures.background.sampler,
				textures.background.view,
				vk::ImageLayout::eGeneral);

		std::vector<vk::WriteDescriptorSet> writeDescriptorSets;

		// Binding 0 : Vertex shader uniform buffer
		writeDescriptorSets.push_back(
			vks::initializers::writeDescriptorSet(
				descriptorSets.background,
				vk::DescriptorType::eUniformBuffer,
				0,
				&uniformBuffer.descriptor));

		// Binding 1 : Color map 
		writeDescriptorSets.push_back(
			vks::initializers::writeDescriptorSet(
				descriptorSets.background,
				vk::DescriptorType::eCombinedImageSampler,
				1,
				&texDescriptor));

		device.updateDescriptorSets(writeDescriptorSets, nullptr);

		// Cube
		descriptorSets.cube = device.allocateDescriptorSets(allocInfo)[0];
		texDescriptor.sampler = textures.cube.sampler;
		texDescriptor.imageView = textures.cube.view;
		writeDescriptorSets[0].dstSet = descriptorSets.cube;
		writeDescriptorSets[1].dstSet = descriptorSets.cube;
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

		// Wire frame rendering pipeline
		std::array<vk::PipelineShaderStageCreateInfo, 2> shaderStages;

		shaderStages[0] = loadShader(getAssetPath() + "shaders/textoverlay/mesh.vert.spv", vk::ShaderStageFlagBits::eVertex);
		shaderStages[1] = loadShader(getAssetPath() + "shaders/textoverlay/mesh.frag.spv", vk::ShaderStageFlagBits::eFragment);

		vk::GraphicsPipelineCreateInfo pipelineCreateInfo =
			vks::initializers::pipelineCreateInfo(
				pipelineLayout,
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

		pipelines.solid = device.createGraphicsPipelines(pipelineCache, pipelineCreateInfo)[0];

		// Background rendering pipeline
		depthStencilState.depthTestEnable = VK_FALSE;
		depthStencilState.depthWriteEnable = VK_FALSE;

		rasterizationState.polygonMode = vk::PolygonMode::eFill;

		shaderStages[0] = loadShader(getAssetPath() + "shaders/textoverlay/background.vert.spv", vk::ShaderStageFlagBits::eVertex);
		shaderStages[1] = loadShader(getAssetPath() + "shaders/textoverlay/background.frag.spv", vk::ShaderStageFlagBits::eFragment);

		pipelines.background = device.createGraphicsPipelines(pipelineCache, pipelineCreateInfo)[0];
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
		uboVS.projection = glm::perspective(glm::radians(60.0f), (float)width / (float)height, 0.1f, 256.0f);

		glm::mat4 viewMatrix = glm::translate(glm::mat4(), glm::vec3(0.0f, 0.0f, zoom));

		uboVS.model = viewMatrix * glm::translate(glm::mat4(), cameraPos);
		uboVS.model = glm::rotate(uboVS.model, glm::radians(rotation.x), glm::vec3(1.0f, 0.0f, 0.0f));
		uboVS.model = glm::rotate(uboVS.model, glm::radians(rotation.y), glm::vec3(0.0f, 1.0f, 0.0f));
		uboVS.model = glm::rotate(uboVS.model, glm::radians(rotation.z), glm::vec3(0.0f, 0.0f, 1.0f));

		memcpy(uniformBuffer.mapped, &uboVS, sizeof(uboVS));
	}

	void prepareTextOverlay()
	{
		// Load the text rendering shaders
		std::vector<vk::PipelineShaderStageCreateInfo> shaderStages;
		shaderStages.push_back(loadShader(getAssetPath() + "shaders/textoverlay/text.vert.spv", vk::ShaderStageFlagBits::eVertex));
		shaderStages.push_back(loadShader(getAssetPath() + "shaders/textoverlay/text.frag.spv", vk::ShaderStageFlagBits::eFragment));

		textOverlay = new TextOverlay(
			vulkanDevice,
			queue,
			frameBuffers,
			swapChain.colorFormat,
			depthFormat,
			&width,
			&height,
			shaderStages
			);
		updateTextOverlay();
	}

	void draw()
	{
		VulkanExampleBase::prepareFrame();

		// Command buffer to be sumitted to the queue
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &drawCmdBuffers[currentBuffer];

		// Submit to queue
		queue.submit(submitInfo, vk::Fence(nullptr));

		// Submit text overlay to queue
		textOverlay->submit(queue, currentBuffer);

		VulkanExampleBase::submitFrame();
	}

	void prepare()
	{
		VulkanExampleBase::prepare();
		loadAssets();
		setupVertexDescriptions();
		prepareUniformBuffers();
		setupDescriptorSetLayout();
		preparePipelines();
		setupDescriptorPool();
		setupDescriptorSet();
		buildCommandBuffers();
		prepareTextOverlay();
		prepared = true;
	}

	virtual void render()
	{
		if (!prepared)
			return;
		draw();
		if (frameCounter == 0)
		{
			vkDeviceWaitIdle(device);
			updateTextOverlay();
		}
	}

	virtual void viewChanged()
	{
		vkDeviceWaitIdle(device);
		updateUniformBuffers();
		updateTextOverlay();
	}

	virtual void windowResized()
	{
		updateTextOverlay();
	}

	virtual void keyPressed(uint32_t keyCode)
	{
		switch (keyCode)
		{
		case KEY_KPADD:
		case KEY_SPACE:
			textOverlay->visible = !textOverlay->visible;
		}
	}
};

VULKAN_EXAMPLE_MAIN()