/*
* Vulkan Example - Texture loading (and display) example (including mip maps)
*
* Copyright (C) 2016-2017 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include "VulkanExampleBase.hpp"
#include "keycodes.hpp"

#include <gli/gli.hpp>

#include <sstream>
#include <iomanip>

#define VERTEX_BUFFER_BIND_ID 0
#define ENABLE_VALIDATION false

// Vertex layout for this example
struct Vertex {
	float pos[3];
	float uv[2];
	float normal[3];
};

class VulkanExample : public VulkanExampleBase
{
public:
	// Contains all Vulkan objects that are required to store and use a texture
	// Note that this repository contains a texture class (VulkanTexture.hpp) that encapsulates texture loading functionality in a class that is used in subsequent demos
	struct Texture {
		vk::Sampler sampler;
		vk::Image image;
		vk::ImageLayout imageLayout;
		vk::DeviceMemory deviceMemory;
		vk::ImageView view;
		uint32_t width, height;
		uint32_t mipLevels;
	} texture;

	struct {
		vk::PipelineVertexInputStateCreateInfo inputState;
		std::vector<vk::VertexInputBindingDescription> bindingDescriptions;
		std::vector<vk::VertexInputAttributeDescription> attributeDescriptions;
	} vertices;

	vks::Buffer vertexBuffer;
	vks::Buffer indexBuffer;
	uint32_t indexCount;

	vks::Buffer uniformBufferVS;

	struct {
		glm::mat4 projection;
		glm::mat4 model;
		glm::vec4 viewPos;
		float lodBias = 0.0f;
	} uboVS;

	struct {
		vk::Pipeline solid;
	} pipelines;

	vk::PipelineLayout pipelineLayout;
	vk::DescriptorSet descriptorSet;
	vk::DescriptorSetLayout descriptorSetLayout;

	VulkanExample() : VulkanExampleBase(ENABLE_VALIDATION)
	{
		zoom = -2.5f;
		rotation = { 0.0f, 15.0f, 0.0f };
		title = "Vulkan Example - Texture loading";
		enableTextOverlay = true;
	}

	~VulkanExample()
	{
		// Clean up used Vulkan resources 
		// Note : Inherited destructor cleans up resources stored in base class

		destroyTextureImage(texture);

		device.destroyPipeline(pipelines.solid);

		device.destroyPipelineLayout(pipelineLayout);
		device.destroyDescriptorSetLayout(descriptorSetLayout);

		vertexBuffer.destroy();
		indexBuffer.destroy();
		uniformBufferVS.destroy();
	}

	// Enable physical device features required for this example				
	virtual void getEnabledFeatures()
	{
		// Enable anisotropic filtering if supported
		if (deviceFeatures.samplerAnisotropy) {
			enabledFeatures.samplerAnisotropy = VK_TRUE;
		};
	}

	// Create an image memory barrier used to change the layout of an image and put it into an active command buffer
	void setImageLayout(vk::CommandBuffer cmdBuffer, vk::Image image, vk::ImageAspectFlags aspectMask, vk::ImageLayout oldImageLayout, vk::ImageLayout newImageLayout, vk::ImageSubresourceRange subresourceRange)
	{
		// Create an image barrier object
		vk::ImageMemoryBarrier imageMemoryBarrier = vks::initializers::imageMemoryBarrier();;
		imageMemoryBarrier.oldLayout = oldImageLayout;
		imageMemoryBarrier.newLayout = newImageLayout;
		imageMemoryBarrier.image = image;
		imageMemoryBarrier.subresourceRange = subresourceRange;

		// Only sets masks for layouts used in this example
		// For a more complete version that can be used with other layouts see vks::tools::setImageLayout

		// Source layouts (old)
		switch (oldImageLayout)
		{
		case vk::ImageLayout::eUndefined:
			// Only valid as initial layout, memory contents are not preserved
			// Can be accessed directly, no source dependency required
			//imageMemoryBarrier.srcAccessMask = 0;
			break;
		case vk::ImageLayout::ePreinitialized:
			// Only valid as initial layout for linear images, preserves memory contents
			// Make sure host writes to the image have been finished
			imageMemoryBarrier.srcAccessMask = vk::AccessFlagBits::eHostWrite;
			break;
		case vk::ImageLayout::eTransferDstOptimal:
			// Old layout is transfer destination
			// Make sure any writes to the image have been finished
			imageMemoryBarrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
			break;
		default:
			break;
		}
		
		// Target layouts (new)
		switch (newImageLayout)
		{
		case vk::ImageLayout::eTransferSrcOptimal:
			// Transfer source (copy, blit)
			// Make sure any reads from the image have been finished
			imageMemoryBarrier.dstAccessMask = vk::AccessFlagBits::eTransferRead;
			break;
		case vk::ImageLayout::eTransferDstOptimal:
			// Transfer destination (copy, blit)
			// Make sure any writes to the image have been finished
			imageMemoryBarrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;
			break;
		case vk::ImageLayout::eShaderReadOnlyOptimal:
			// Shader read (sampler, input attachment)
			imageMemoryBarrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;
			break;
		default:
			break;
		}

		// Put barrier on top of pipeline
		vk::PipelineStageFlags srcStageFlags = vk::PipelineStageFlagBits::eTopOfPipe;
		vk::PipelineStageFlags destStageFlags = vk::PipelineStageFlagBits::eTopOfPipe;

		// Put barrier inside setup command buffer
		cmdBuffer.pipelineBarrier(
			srcStageFlags, 
			destStageFlags, 
			vk::DependencyFlags(), 
			nullptr,
			nullptr,
			imageMemoryBarrier);
	}

	void loadTexture()
	{
		// We use the Khronos texture format (https://www.khronos.org/opengles/sdk/tools/KTX/file_format_spec/) 
		std::string filename = getAssetPath() + "textures/metalplate01_rgba.ktx";
		// Texture data contains 4 channels (RGBA) with unnormalized 8-bit values, this is the most commonly supported format
		vk::Format format = vk::Format::eR8G8B8A8Unorm;

		// Set to true to use linear tiled images 
		// This is just for learning purposes and not suggested, as linear tiled images are pretty restricted and often only support a small set of features (e.g. no mips, etc.)
		bool forceLinearTiling = false;

#if defined(__ANDROID__)
		// Textures are stored inside the apk on Android (compressed)
		// So they need to be loaded via the asset manager
		AAsset* asset = AAssetManager_open(androidApp->activity->assetManager, filename.c_str(), AASSET_MODE_STREAMING);
		assert(asset);
		size_t size = AAsset_getLength(asset);
		assert(size > 0);

		void *textureData = malloc(size);
		AAsset_read(asset, textureData, size);
		AAsset_close(asset);

		gli::texture2d tex2D(gli::load((const char*)textureData, size));
#else
		gli::texture2d tex2D(gli::load(filename));
#endif

		assert(!tex2D.empty());

		vk::FormatProperties formatProperties;

		texture.width = static_cast<uint32_t>(tex2D[0].extent().x);
		texture.height = static_cast<uint32_t>(tex2D[0].extent().y);
		texture.mipLevels = static_cast<uint32_t>(tex2D.levels());

		// Get device properites for the requested texture format
		formatProperties = physicalDevice.getFormatProperties(format);

		// Only use linear tiling if requested (and supported by the device)
		// Support for linear tiling is mostly limited, so prefer to use
		// optimal tiling instead
		// On most implementations linear tiling will only support a very
		// limited amount of formats and features (mip maps, cubemaps, arrays, etc.)
		vk::Bool32 useStaging = true;

		// Only use linear tiling if forced
		if (forceLinearTiling)
		{
			// Don't use linear if format is not supported for (linear) shader sampling
			useStaging = !(formatProperties.linearTilingFeatures & vk::FormatFeatureFlagBits::eSampledImage);
		}

		vk::MemoryAllocateInfo memAllocInfo = vks::initializers::memoryAllocateInfo();
		vk::MemoryRequirements memReqs = {};

		if (useStaging)
		{
			// Create a host-visible staging buffer that contains the raw image data
			vk::Buffer stagingBuffer;
			vk::DeviceMemory stagingMemory;

			vk::BufferCreateInfo bufferCreateInfo = vks::initializers::bufferCreateInfo();
			bufferCreateInfo.size = tex2D.size();
			// This buffer is used as a transfer source for the buffer copy
			bufferCreateInfo.usage = vk::BufferUsageFlagBits::eTransferSrc;
			bufferCreateInfo.sharingMode = vk::SharingMode::eExclusive;
			
			stagingBuffer = device.createBuffer(bufferCreateInfo);

			// Get memory requirements for the staging buffer (alignment, memory type bits)
			memReqs = device.getBufferMemoryRequirements(stagingBuffer);

			memAllocInfo.allocationSize = memReqs.size;
			// Get memory type index for a host visible buffer
			memAllocInfo.memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

			stagingMemory = device.allocateMemory(memAllocInfo);
			device.bindBufferMemory(stagingBuffer, stagingMemory, 0);

			// Copy texture data into staging buffer
			uint8_t *data = (uint8_t*)device.mapMemory(stagingMemory, 0, memReqs.size, vk::MemoryMapFlags());
			memcpy(data, tex2D.data(), tex2D.size());
			device.unmapMemory(stagingMemory);

			// Setup buffer copy regions for each mip level
			std::vector<vk::BufferImageCopy> bufferCopyRegions;
			uint32_t offset = 0;

			for (uint32_t i = 0; i < texture.mipLevels; i++)
			{
				vk::BufferImageCopy bufferCopyRegion = {};
				bufferCopyRegion.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
				bufferCopyRegion.imageSubresource.mipLevel = i;
				bufferCopyRegion.imageSubresource.baseArrayLayer = 0;
				bufferCopyRegion.imageSubresource.layerCount = 1;
				bufferCopyRegion.imageExtent.width = static_cast<uint32_t>(tex2D[i].extent().x);
				bufferCopyRegion.imageExtent.height = static_cast<uint32_t>(tex2D[i].extent().y);
				bufferCopyRegion.imageExtent.depth = 1;
				bufferCopyRegion.bufferOffset = offset;

				bufferCopyRegions.push_back(bufferCopyRegion);

				offset += static_cast<uint32_t>(tex2D[i].size());
			}

			// Create optimal tiled target image
			vk::ImageCreateInfo imageCreateInfo = vks::initializers::imageCreateInfo();
			imageCreateInfo.imageType = vk::ImageType::e2D;
			imageCreateInfo.format = format;
			imageCreateInfo.mipLevels = texture.mipLevels;
			imageCreateInfo.arrayLayers = 1;
			imageCreateInfo.samples = vk::SampleCountFlagBits::e1;
			imageCreateInfo.tiling = vk::ImageTiling::eOptimal;
			imageCreateInfo.sharingMode = vk::SharingMode::eExclusive;
			// Set initial layout of the image to undefined
			imageCreateInfo.initialLayout = vk::ImageLayout::eUndefined;
			imageCreateInfo.extent = vk::Extent3D{ texture.width, texture.height, 1 };
			imageCreateInfo.usage = vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled;

			texture.image = device.createImage(imageCreateInfo);

			memReqs = device.getImageMemoryRequirements(texture.image);

			memAllocInfo.allocationSize = memReqs.size;
			memAllocInfo.memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);

			texture.deviceMemory = device.allocateMemory(memAllocInfo);
			device.bindImageMemory(texture.image, texture.deviceMemory, 0);

			vk::CommandBuffer copyCmd = VulkanExampleBase::createCommandBuffer(vk::CommandBufferLevel::ePrimary, true);

			// Image barrier for optimal image

			// The sub resource range describes the regions of the image we will be transition
			vk::ImageSubresourceRange subresourceRange;
			// Image only contains color data
			subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
			// Start at first mip level
			subresourceRange.baseMipLevel = 0;
			// We will transition on all mip levels
			subresourceRange.levelCount = texture.mipLevels;
			// The 2D texture only has one layer
			subresourceRange.layerCount = 1;

			// Optimal image will be used as destination for the copy, so we must transfer from our
			// initial undefined image layout to the transfer destination layout
			setImageLayout(
				copyCmd,
				texture.image,
				vk::ImageAspectFlagBits::eColor,
				vk::ImageLayout::eUndefined,
				vk::ImageLayout::eTransferDstOptimal,
				subresourceRange);

			// Copy mip levels from staging buffer
			copyCmd.copyBufferToImage(
				stagingBuffer,
				texture.image,
				vk::ImageLayout::eTransferDstOptimal,
				bufferCopyRegions);

			// Change texture image layout to shader read after all mip levels have been copied
			texture.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
			setImageLayout(
				copyCmd,
				texture.image,
				vk::ImageAspectFlagBits::eColor,
				vk::ImageLayout::eTransferDstOptimal,
				texture.imageLayout,
				subresourceRange);

			VulkanExampleBase::flushCommandBuffer(copyCmd, queue, true);

			// Clean up staging resources
			device.freeMemory(stagingMemory);
			device.destroyBuffer(stagingBuffer);
		}
		else
		{
			vk::Image mappableImage;
			vk::DeviceMemory mappableMemory;

			// Load mip map level 0 to linear tiling image
			vk::ImageCreateInfo imageCreateInfo = vks::initializers::imageCreateInfo();
			imageCreateInfo.imageType = vk::ImageType::e2D;
			imageCreateInfo.format = format;
			imageCreateInfo.mipLevels = 1;
			imageCreateInfo.arrayLayers = 1;
			imageCreateInfo.samples = vk::SampleCountFlagBits::e1;
			imageCreateInfo.tiling = vk::ImageTiling::eLinear;
			imageCreateInfo.usage = vk::ImageUsageFlagBits::eSampled;
			imageCreateInfo.sharingMode = vk::SharingMode::eExclusive;
			imageCreateInfo.initialLayout = vk::ImageLayout::ePreinitialized;
			imageCreateInfo.extent = vk::Extent3D{ texture.width, texture.height, 1 };
			mappableImage = device.createImage(imageCreateInfo);

			// Get memory requirements for this image 
			// like size and alignment
			memReqs = device.getImageMemoryRequirements(mappableImage);
			// Set memory allocation size to required memory size
			memAllocInfo.allocationSize = memReqs.size;

			// Get memory type that can be mapped to host memory
			memAllocInfo.memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

			// Allocate host memory
			mappableMemory = device.allocateMemory(memAllocInfo);

			// Bind allocated image for use
			device.bindImageMemory(mappableImage, mappableMemory, 0);

			// Get sub resource layout
			// Mip map count, array layer, etc.
			vk::ImageSubresource subRes = {};
			subRes.aspectMask = vk::ImageAspectFlagBits::eColor;

			vk::SubresourceLayout subResLayout;
			void *data;

			// Get sub resources layout 
			// Includes row pitch, size offsets, etc.
			subResLayout = device.getImageSubresourceLayout(mappableImage, subRes);

			// Map image memory
			data = device.mapMemory(mappableMemory, 0, memReqs.size, vk::MemoryMapFlags());

			// Copy image data into memory
			memcpy(data, tex2D[subRes.mipLevel].data(), tex2D[subRes.mipLevel].size());

			device.unmapMemory(mappableMemory);

			// Linear tiled images don't need to be staged
			// and can be directly used as textures
			texture.image = mappableImage;
			texture.deviceMemory = mappableMemory;
			texture.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
			
			vk::CommandBuffer copyCmd = VulkanExampleBase::createCommandBuffer(vk::CommandBufferLevel::ePrimary, true);

			// Setup image memory barrier transfer image to shader read layout

			// The sub resource range describes the regions of the image we will be transition
			vk::ImageSubresourceRange subresourceRange;
			// Image only contains color data
			subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
			// Start at first mip level
			subresourceRange.baseMipLevel = 0;
			// Only one mip level, most implementations won't support more for linear tiled images
			subresourceRange.levelCount = 1;
			// The 2D texture only has one layer
			subresourceRange.layerCount = 1;

			setImageLayout(
				copyCmd, 
				texture.image,
				vk::ImageAspectFlagBits::eColor, 
				vk::ImageLayout::ePreinitialized,
				texture.imageLayout,
				subresourceRange);

			VulkanExampleBase::flushCommandBuffer(copyCmd, queue, true);
		}

		// Create a texture sampler
		// In Vulkan textures are accessed by samplers
		// This separates all the sampling information from the texture data. This means you could have multiple sampler objects for the same texture with different settings
		// Note: Similar to the samplers available with OpenGL 3.3
		vk::SamplerCreateInfo sampler = vks::initializers::samplerCreateInfo();
		sampler.magFilter = vk::Filter::eLinear;
		sampler.minFilter = vk::Filter::eLinear;
		sampler.mipmapMode = vk::SamplerMipmapMode::eLinear;
		sampler.addressModeU = vk::SamplerAddressMode::eRepeat;
		sampler.addressModeV = vk::SamplerAddressMode::eRepeat;
		sampler.addressModeW = vk::SamplerAddressMode::eRepeat;
		sampler.mipLodBias = 0.0f;
		sampler.compareOp = vk::CompareOp::eNever;
		sampler.minLod = 0.0f;
		// Set max level-of-detail to mip level count of the texture
		sampler.maxLod = (useStaging) ? (float)texture.mipLevels : 0.0f;
		// Enable anisotropic filtering
		// This feature is optional, so we must check if it's supported on the device
		if (vulkanDevice->features.samplerAnisotropy)
		{
			// Use max. level of anisotropy for this example
			sampler.maxAnisotropy = vulkanDevice->properties.limits.maxSamplerAnisotropy;
			sampler.anisotropyEnable = VK_TRUE;
		}
		else
		{
			// The device does not support anisotropic filtering
			sampler.maxAnisotropy = 1.0;
			sampler.anisotropyEnable = VK_FALSE;
		}
		sampler.borderColor = vk::BorderColor::eFloatOpaqueWhite;
		texture.sampler = device.createSampler(sampler);

		// Create image view
		// Textures are not directly accessed by the shaders and
		// are abstracted by image views containing additional
		// information and sub resource ranges
		vk::ImageViewCreateInfo view = vks::initializers::imageViewCreateInfo();
		view.viewType = vk::ImageViewType::e2D;
		view.format = format;
		view.components = { vk::ComponentSwizzle::eR, vk::ComponentSwizzle::eG, vk::ComponentSwizzle::eB, vk::ComponentSwizzle::eA };
		// The subresource range describes the set of mip levels (and array layers) that can be accessed through this image view
		// It's possible to create multiple image views for a single image referring to different (and/or overlapping) ranges of the image
		view.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
		view.subresourceRange.baseMipLevel = 0;
		view.subresourceRange.baseArrayLayer = 0;
		view.subresourceRange.layerCount = 1;
		// Linear tiling usually won't support mip maps
		// Only set mip map count if optimal tiling is used
		view.subresourceRange.levelCount = (useStaging) ? texture.mipLevels : 1;
		// The view will be based on the texture's image
		view.image = texture.image;
		texture.view = device.createImageView(view);
	}

	// Free all Vulkan resources used by a texture object
	void destroyTextureImage(Texture texture)
	{
		device.destroyImageView(texture.view);
		device.destroyImage(texture.image);
		device.destroySampler(texture.sampler);
		device.freeMemory(texture.deviceMemory);
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

			drawCmdBuffers[i].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSet, nullptr);
			drawCmdBuffers[i].bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.solid);

			std::vector<vk::DeviceSize> offsets = { 0 };
			drawCmdBuffers[i].bindVertexBuffers(VERTEX_BUFFER_BIND_ID, vertexBuffer.buffer, offsets);
			drawCmdBuffers[i].bindIndexBuffer(indexBuffer.buffer, 0, vk::IndexType::eUint32);

			drawCmdBuffers[i].drawIndexed(indexCount, 1, 0, 0, 0);

			drawCmdBuffers[i].endRenderPass();

			drawCmdBuffers[i].end();
		}
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

	void generateQuad()
	{
		// Setup vertices for a single uv-mapped quad made from two triangles
		std::vector<Vertex> vertices =
		{
			{ {  1.0f,  1.0f, 0.0f }, { 1.0f, 1.0f },{ 0.0f, 0.0f, 1.0f } },
			{ { -1.0f,  1.0f, 0.0f }, { 0.0f, 1.0f },{ 0.0f, 0.0f, 1.0f } },
			{ { -1.0f, -1.0f, 0.0f }, { 0.0f, 0.0f },{ 0.0f, 0.0f, 1.0f } },
			{ {  1.0f, -1.0f, 0.0f }, { 1.0f, 0.0f },{ 0.0f, 0.0f, 1.0f } }
		};

		// Setup indices
		std::vector<uint32_t> indices = { 0,1,2, 2,3,0 };
		indexCount = static_cast<uint32_t>(indices.size());

		// Create buffers
		// For the sake of simplicity we won't stage the vertex data to the gpu memory
		// Vertex buffer
		vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eVertexBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			&vertexBuffer,
			vertices.size() * sizeof(Vertex),
			vertices.data());
		// Index buffer
		vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eIndexBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			&indexBuffer,
			indices.size() * sizeof(uint32_t),
			indices.data());
	}

	void setupVertexDescriptions()
	{
		// Binding description
		vertices.bindingDescriptions.resize(1);
		vertices.bindingDescriptions[0] =
			vks::initializers::vertexInputBindingDescription(
				VERTEX_BUFFER_BIND_ID, 
				sizeof(Vertex), 
				vk::VertexInputRate::eVertex);

		// Attribute descriptions
		// Describes memory layout and shader positions
		vertices.attributeDescriptions.resize(3);
		// Location 0 : Position
		vertices.attributeDescriptions[0] =
			vks::initializers::vertexInputAttributeDescription(
				VERTEX_BUFFER_BIND_ID,
				0,
				vk::Format::eR32G32B32Sfloat,
				offsetof(Vertex, pos));			
		// Location 1 : Texture coordinates
		vertices.attributeDescriptions[1] =
			vks::initializers::vertexInputAttributeDescription(
				VERTEX_BUFFER_BIND_ID,
				1,
				vk::Format::eR32G32Sfloat,
				offsetof(Vertex, uv));
		// Location 1 : Vertex normal
		vertices.attributeDescriptions[2] =
			vks::initializers::vertexInputAttributeDescription(
				VERTEX_BUFFER_BIND_ID,
				2,
				vk::Format::eR32G32B32Sfloat,
				offsetof(Vertex, normal));

		vertices.inputState = vks::initializers::pipelineVertexInputStateCreateInfo();
		vertices.inputState.vertexBindingDescriptionCount = static_cast<uint32_t>(vertices.bindingDescriptions.size());
		vertices.inputState.pVertexBindingDescriptions = vertices.bindingDescriptions.data();
		vertices.inputState.vertexAttributeDescriptionCount = static_cast<uint32_t>(vertices.attributeDescriptions.size());
		vertices.inputState.pVertexAttributeDescriptions = vertices.attributeDescriptions.data();
	}

	void setupDescriptorPool()
	{
		// Example uses one ubo and one image sampler
		std::vector<vk::DescriptorPoolSize> poolSizes =
		{
			vks::initializers::descriptorPoolSize(vk::DescriptorType::eUniformBuffer, 1),
			vks::initializers::descriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, 1)
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
			// Binding 1 : Fragment shader image sampler
			vks::initializers::descriptorSetLayoutBinding(
				vk::DescriptorType::eCombinedImageSampler, 
				vk::ShaderStageFlagBits::eFragment, 
				1)
		};

		vk::DescriptorSetLayoutCreateInfo descriptorLayout = 
			vks::initializers::descriptorSetLayoutCreateInfo(
				setLayoutBindings);

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

		// Setup a descriptor image info for the current texture to be used as a combined image sampler
		vk::DescriptorImageInfo textureDescriptor;
		textureDescriptor.imageView = texture.view;				// The image's view (images are never directly accessed by the shader, but rather through views defining subresources)
		textureDescriptor.sampler = texture.sampler;			// The sampler (Telling the pipeline how to sample the texture, including repeat, border, etc.)
		textureDescriptor.imageLayout = texture.imageLayout;	// The current layout of the image (Note: Should always fit the actual use, e.g. shader read)

		std::vector<vk::WriteDescriptorSet> writeDescriptorSets =
		{
			// Binding 0 : Vertex shader uniform buffer
			vks::initializers::writeDescriptorSet(
				descriptorSet, 
				vk::DescriptorType::eUniformBuffer, 
				0, 
				&uniformBufferVS.descriptor),
			// Binding 1 : Fragment shader texture sampler
			//	Fragment shader: layout (binding = 1) uniform sampler2D samplerColor;
			vks::initializers::writeDescriptorSet(
				descriptorSet, 				
				vk::DescriptorType::eCombinedImageSampler,		// The descriptor set will use a combined image sampler (sampler and image could be split)
				1,												// Shader binding point 1
				&textureDescriptor)								// Pointer to the descriptor image for our texture
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

		// Load shaders
		std::array<vk::PipelineShaderStageCreateInfo,2> shaderStages;

		shaderStages[0] = loadShader(getAssetPath() + "shaders/texture/texture.vert.spv", vk::ShaderStageFlagBits::eVertex);
		shaderStages[1] = loadShader(getAssetPath() + "shaders/texture/texture.frag.spv", vk::ShaderStageFlagBits::eFragment);

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
	}

	// Prepare and initialize uniform buffer containing shader uniforms
	void prepareUniformBuffers()
	{
		// Vertex shader uniform buffer block
		vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eUniformBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			&uniformBufferVS,
			sizeof(uboVS),
			&uboVS);

		updateUniformBuffers();
	}

	void updateUniformBuffers()
	{
		// Vertex shader
		uboVS.projection = glm::perspective(glm::radians(60.0f), (float)width / (float)height, 0.001f, 256.0f);
		glm::mat4 viewMatrix = glm::translate(glm::mat4(), glm::vec3(0.0f, 0.0f, zoom));

		uboVS.model = viewMatrix * glm::translate(glm::mat4(), cameraPos);
		uboVS.model = glm::rotate(uboVS.model, glm::radians(rotation.x), glm::vec3(1.0f, 0.0f, 0.0f));
		uboVS.model = glm::rotate(uboVS.model, glm::radians(rotation.y), glm::vec3(0.0f, 1.0f, 0.0f));
		uboVS.model = glm::rotate(uboVS.model, glm::radians(rotation.z), glm::vec3(0.0f, 0.0f, 1.0f));

		uboVS.viewPos = glm::vec4(0.0f, 0.0f, -zoom, 0.0f);

		uniformBufferVS.map();
		memcpy(uniformBufferVS.mapped, &uboVS, sizeof(uboVS));
		uniformBufferVS.unmap();
	}

	void prepare()
	{
		VulkanExampleBase::prepare();
		loadTexture();
		generateQuad();
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

	void changeLodBias(float delta)
	{
		uboVS.lodBias += delta;
		if (uboVS.lodBias < 0.0f)
		{
			uboVS.lodBias = 0.0f;
		}
		if (uboVS.lodBias > texture.mipLevels)
		{
			uboVS.lodBias = (float)texture.mipLevels;
		}
		updateUniformBuffers();
		updateTextOverlay();
	}

	virtual void keyPressed(uint32_t keyCode)
	{
		switch (keyCode)
		{
		case KEY_KPADD:
		case GAMEPAD_BUTTON_R1:
			changeLodBias(0.1f);
			break;
		case KEY_KPSUB:
		case GAMEPAD_BUTTON_L1:
			changeLodBias(-0.1f);
			break;
		}
	}

	virtual void getOverlayText(VulkanTextOverlay *textOverlay)
	{
		if (vulkanDevice->features.samplerAnisotropy) {
			std::stringstream ss;
			ss << std::setprecision(2) << std::fixed << uboVS.lodBias;
#if defined(__ANDROID__)
			textOverlay->addText("LOD bias: " + ss.str() + " (Buttons L1/R1 to change)", 5.0f, 85.0f, VulkanTextOverlay::alignLeft);
#else
			textOverlay->addText("LOD bias: " + ss.str() + " (numpad +/- to change)", 5.0f, 85.0f, VulkanTextOverlay::alignLeft);
#endif
		}
	}
};

VULKAN_EXAMPLE_MAIN()
