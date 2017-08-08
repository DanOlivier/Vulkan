/*
* Vulkan texture loader
*
* Copyright(C) 2016-2017 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license(MIT) (http://opensource.org/licenses/MIT)
*/

#pragma once

#include <vector>

#include "vulkan/vulkan.hpp"

#include <gli/gli.hpp>

#include "VulkanTools.h"
#include "VulkanDevice.hpp"

#if defined(__ANDROID__)
#include <android/asset_manager.h>
#endif

namespace vks
{
	/** @brief Vulkan texture base class */
	class Texture {
	public:
		vks::VulkanDevice *device;
		vk::Image image;
		vk::ImageLayout imageLayout;
		vk::DeviceMemory deviceMemory;
		vk::ImageView view;
		uint32_t width, height;
		uint32_t mipLevels;
		uint32_t layerCount;
		vk::DescriptorImageInfo descriptor;

		/** @brief Optional sampler to use with this texture */
		vk::Sampler sampler;

		/** @brief Update image descriptor from current sampler, view and image layout */
		void updateDescriptor()
		{
			descriptor.sampler = sampler;
			descriptor.imageView = view;
			descriptor.imageLayout = imageLayout;
		}

		/** @brief Release all Vulkan resources held by this texture */
		void destroy()
		{
			device->logicalDevice.destroyImageView(view);
			device->logicalDevice.destroyImage(image);
			if (sampler)
			{
				device->logicalDevice.destroySampler(sampler);
			}
			device->logicalDevice.freeMemory(deviceMemory);
		}
	};

	/** @brief 2D texture */
	class Texture2D : public Texture {
	public:
		/**
		* Load a 2D texture including all mip levels
		*
		* @param filename File to load (supports .ktx and .dds)
		* @param format Vulkan format of the image data stored in the file
		* @param device Vulkan device to create the texture on
		* @param copyQueue Queue used for the texture staging copy commands (must support transfer)
		* @param (Optional) imageUsageFlags Usage flags for the texture's image (defaults to vk::ImageUsageFlagBits::eSampled)
		* @param (Optional) imageLayout Usage layout for the texture (defaults vk::ImageLayout::eShaderReadOnlyOptimal)
		* @param (Optional) forceLinear Force linear tiling (not advised, defaults to false)
		*
		*/
		void loadFromFile(
			std::string filename, 
			vk::Format format,
			vks::VulkanDevice *device,
			vk::Queue copyQueue,
			vk::ImageUsageFlags imageUsageFlags = vk::ImageUsageFlagBits::eSampled,
			vk::ImageLayout imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal, 
			bool forceLinear = false)
		{
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

			free(textureData);
#else
			if (!vks::tools::fileExists(filename)) {
				vks::tools::exitFatal("Could not load texture from " + filename, "File not found");
			}
			gli::texture2d tex2D(gli::load(filename.c_str()));
#endif		
			assert(!tex2D.empty());

			this->device = device;
			width = static_cast<uint32_t>(tex2D[0].extent().x);
			height = static_cast<uint32_t>(tex2D[0].extent().y);
			mipLevels = static_cast<uint32_t>(tex2D.levels());

			// Get device properites for the requested texture format
			vk::FormatProperties formatProperties;
			formatProperties = device->physicalDevice.getFormatProperties(format);

			// Only use linear tiling if requested (and supported by the device)
			// Support for linear tiling is mostly limited, so prefer to use
			// optimal tiling instead
			// On most implementations linear tiling will only support a very
			// limited amount of formats and features (mip maps, cubemaps, arrays, etc.)
			vk::Bool32 useStaging = !forceLinear;

			vk::MemoryAllocateInfo memAllocInfo = vks::initializers::memoryAllocateInfo();
			vk::MemoryRequirements memReqs;

			// Use a separate command buffer for texture loading
			vk::CommandBuffer copyCmd = device->createCommandBuffer(vk::CommandBufferLevel::ePrimary, true);

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

				stagingBuffer = device->logicalDevice.createBuffer(bufferCreateInfo);

				// Get memory requirements for the staging buffer (alignment, memory type bits)
				memReqs = device->logicalDevice.getBufferMemoryRequirements(stagingBuffer);

				memAllocInfo.allocationSize = memReqs.size;
				// Get memory type index for a host visible buffer
				memAllocInfo.memoryTypeIndex = device->getMemoryType(memReqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

				stagingMemory = device->logicalDevice.allocateMemory(memAllocInfo);
				device->logicalDevice.bindBufferMemory(stagingBuffer, stagingMemory, 0);

				// Copy texture data into staging buffer
				uint8_t *data = (uint8_t*)device->logicalDevice.mapMemory(stagingMemory, 0, memReqs.size, vk::MemoryMapFlags());
				memcpy(data, tex2D.data(), tex2D.size());
				device->logicalDevice.unmapMemory(stagingMemory);

				// Setup buffer copy regions for each mip level
				std::vector<vk::BufferImageCopy> bufferCopyRegions;
				uint32_t offset = 0;

				for (uint32_t i = 0; i < mipLevels; i++)
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
				imageCreateInfo.mipLevels = mipLevels;
				imageCreateInfo.arrayLayers = 1;
				imageCreateInfo.samples = vk::SampleCountFlagBits::e1;
				imageCreateInfo.tiling = vk::ImageTiling::eOptimal;
				imageCreateInfo.sharingMode = vk::SharingMode::eExclusive;
				imageCreateInfo.initialLayout = vk::ImageLayout::eUndefined;
				imageCreateInfo.extent = vk::Extent3D{ width, height, 1 };
				imageCreateInfo.usage = imageUsageFlags;
				// Ensure that the TRANSFER_DST bit is set for staging
				if (!(imageCreateInfo.usage & vk::ImageUsageFlagBits::eTransferDst))
				{
					imageCreateInfo.usage |= vk::ImageUsageFlagBits::eTransferDst;
				}
				image = device->logicalDevice.createImage(imageCreateInfo);

				memReqs = device->logicalDevice.getImageMemoryRequirements(image);

				memAllocInfo.allocationSize = memReqs.size;

				memAllocInfo.memoryTypeIndex = device->getMemoryType(memReqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);
				deviceMemory = device->logicalDevice.allocateMemory(memAllocInfo);
				device->logicalDevice.bindImageMemory(image, deviceMemory, 0);

				vk::ImageSubresourceRange subresourceRange;
				subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
				subresourceRange.baseMipLevel = 0;
				subresourceRange.levelCount = mipLevels;
				subresourceRange.layerCount = 1;

				// Image barrier for optimal image (target)
				// Optimal image will be used as destination for the copy
				vks::tools::setImageLayout(
					copyCmd,
					image,
					vk::ImageLayout::eUndefined,
					vk::ImageLayout::eTransferDstOptimal,
					subresourceRange);

				// Copy mip levels from staging buffer
				copyCmd.copyBufferToImage(
					stagingBuffer,
					image,
					vk::ImageLayout::eTransferDstOptimal,
					bufferCopyRegions
				);

				// Change texture image layout to shader read after all mip levels have been copied
				this->imageLayout = imageLayout;
				vks::tools::setImageLayout(
					copyCmd,
					image,
					vk::ImageLayout::eTransferDstOptimal,
					imageLayout,
					subresourceRange);

				device->flushCommandBuffer(copyCmd, copyQueue);

				// Clean up staging resources
				device->logicalDevice.freeMemory(stagingMemory);
				device->logicalDevice.destroyBuffer(stagingBuffer);
			}
			else
			{
				// Prefer using optimal tiling, as linear tiling 
				// may support only a small set of features 
				// depending on implementation (e.g. no mip maps, only one layer, etc.)

				// Check if this support is supported for linear tiling
				assert(formatProperties.linearTilingFeatures & vk::FormatFeatureFlagBits::eSampledImage);

				vk::Image mappableImage;
				vk::DeviceMemory mappableMemory;

				vk::ImageCreateInfo imageCreateInfo = vks::initializers::imageCreateInfo();
				imageCreateInfo.imageType = vk::ImageType::e2D;
				imageCreateInfo.format = format;
				imageCreateInfo.extent = vk::Extent3D{ width, height, 1 };
				imageCreateInfo.mipLevels = 1;
				imageCreateInfo.arrayLayers = 1;
				imageCreateInfo.samples = vk::SampleCountFlagBits::e1;
				imageCreateInfo.tiling = vk::ImageTiling::eLinear;
				imageCreateInfo.usage = imageUsageFlags;
				imageCreateInfo.sharingMode = vk::SharingMode::eExclusive;
				imageCreateInfo.initialLayout = vk::ImageLayout::eUndefined;

				// Load mip map level 0 to linear tiling image
				mappableImage = device->logicalDevice.createImage(imageCreateInfo);

				// Get memory requirements for this image 
				// like size and alignment
				memReqs = device->logicalDevice.getImageMemoryRequirements(mappableImage);
				// Set memory allocation size to required memory size
				memAllocInfo.allocationSize = memReqs.size;

				// Get memory type that can be mapped to host memory
				memAllocInfo.memoryTypeIndex = device->getMemoryType(memReqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

				// Allocate host memory
				mappableMemory = device->logicalDevice.allocateMemory(memAllocInfo);

				// Bind allocated image for use
				device->logicalDevice.bindImageMemory(mappableImage, mappableMemory, 0);

				// Get sub resource layout
				// Mip map count, array layer, etc.
				vk::ImageSubresource subRes = {};
				subRes.aspectMask = vk::ImageAspectFlagBits::eColor;
				subRes.mipLevel = 0;

				vk::SubresourceLayout subResLayout;
				void *data;

				// Get sub resources layout 
				// Includes row pitch, size offsets, etc.
				subResLayout = device->logicalDevice.getImageSubresourceLayout(mappableImage, subRes);

				// Map image memory
				data = device->logicalDevice.mapMemory(mappableMemory, 0, memReqs.size, vk::MemoryMapFlags());

				// Copy image data into memory
				memcpy(data, tex2D[subRes.mipLevel].data(), tex2D[subRes.mipLevel].size());

				device->logicalDevice.unmapMemory(mappableMemory);

				// Linear tiled images don't need to be staged
				// and can be directly used as textures
				image = mappableImage;
				deviceMemory = mappableMemory;
				imageLayout = imageLayout;

				// Setup image memory barrier
				vks::tools::setImageLayout(copyCmd, image, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eUndefined, imageLayout);

				device->flushCommandBuffer(copyCmd, copyQueue);
			}

			// Create a defaultsampler
			vk::SamplerCreateInfo samplerCreateInfo = {};

			samplerCreateInfo.magFilter = vk::Filter::eLinear;
			samplerCreateInfo.minFilter = vk::Filter::eLinear;
			samplerCreateInfo.mipmapMode = vk::SamplerMipmapMode::eLinear;
			samplerCreateInfo.addressModeU = vk::SamplerAddressMode::eRepeat;
			samplerCreateInfo.addressModeV = vk::SamplerAddressMode::eRepeat;
			samplerCreateInfo.addressModeW = vk::SamplerAddressMode::eRepeat;
			samplerCreateInfo.mipLodBias = 0.0f;
			samplerCreateInfo.compareOp = vk::CompareOp::eNever;
			samplerCreateInfo.minLod = 0.0f;
			// Max level-of-detail should match mip level count
			samplerCreateInfo.maxLod = (useStaging) ? (float)mipLevels : 0.0f;
			// Only enable anisotropic filtering if enabled on the devicec
			samplerCreateInfo.maxAnisotropy = device->enabledFeatures.samplerAnisotropy ? device->properties.limits.maxSamplerAnisotropy : 1.0f;
			samplerCreateInfo.anisotropyEnable = VK_TRUE;
			samplerCreateInfo.borderColor = vk::BorderColor::eFloatOpaqueWhite;
			sampler = device->logicalDevice.createSampler(samplerCreateInfo);

			// Create image view
			// Textures are not directly accessed by the shaders and
			// are abstracted by image views containing additional
			// information and sub resource ranges
			vk::ImageViewCreateInfo viewCreateInfo = {};

			viewCreateInfo.viewType = vk::ImageViewType::e2D;
			viewCreateInfo.format = format;
			viewCreateInfo.components = { vk::ComponentSwizzle::eR, vk::ComponentSwizzle::eG, vk::ComponentSwizzle::eB, vk::ComponentSwizzle::eA };
			viewCreateInfo.subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 };
			// Linear tiling usually won't support mip maps
			// Only set mip map count if optimal tiling is used
			viewCreateInfo.subresourceRange.levelCount = (useStaging) ? mipLevels : 1;
			viewCreateInfo.image = image;
			view = device->logicalDevice.createImageView(viewCreateInfo);

			// Update descriptor image info member that can be used for setting up descriptor sets
			updateDescriptor();
		}

		/**
		* Creates a 2D texture from a buffer
		*
		* @param buffer Buffer containing texture data to upload
		* @param bufferSize Size of the buffer in machine units
		* @param width Width of the texture to create
		* @param height Height of the texture to create
		* @param format Vulkan format of the image data stored in the file
		* @param device Vulkan device to create the texture on
		* @param copyQueue Queue used for the texture staging copy commands (must support transfer)
		* @param (Optional) filter Texture filtering for the sampler (defaults to vk::Filter::eLinear)
		* @param (Optional) imageUsageFlags Usage flags for the texture's image (defaults to vk::ImageUsageFlagBits::eSampled)
		* @param (Optional) imageLayout Usage layout for the texture (defaults vk::ImageLayout::eShaderReadOnlyOptimal)
		*/
		void fromBuffer(
			void* buffer,
			vk::DeviceSize bufferSize,
			vk::Format format,
			uint32_t width,
			uint32_t height,
			vks::VulkanDevice *device,
			vk::Queue copyQueue,
			vk::Filter filter = vk::Filter::eLinear,
			vk::ImageUsageFlags imageUsageFlags = vk::ImageUsageFlagBits::eSampled,
			vk::ImageLayout imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal)
		{
			assert(buffer);

			this->device = device;
			width = width;
			height = height;
			mipLevels = 1;

			vk::MemoryAllocateInfo memAllocInfo = vks::initializers::memoryAllocateInfo();
			vk::MemoryRequirements memReqs;

			// Use a separate command buffer for texture loading
			vk::CommandBuffer copyCmd = device->createCommandBuffer(vk::CommandBufferLevel::ePrimary, true);

			// Create a host-visible staging buffer that contains the raw image data
			vk::Buffer stagingBuffer;
			vk::DeviceMemory stagingMemory;

			vk::BufferCreateInfo bufferCreateInfo = vks::initializers::bufferCreateInfo();
			bufferCreateInfo.size = bufferSize;
			// This buffer is used as a transfer source for the buffer copy
			bufferCreateInfo.usage = vk::BufferUsageFlagBits::eTransferSrc;
			bufferCreateInfo.sharingMode = vk::SharingMode::eExclusive;

			stagingBuffer = device->logicalDevice.createBuffer(bufferCreateInfo);

			// Get memory requirements for the staging buffer (alignment, memory type bits)
			memReqs = device->logicalDevice.getBufferMemoryRequirements(stagingBuffer);

			memAllocInfo.allocationSize = memReqs.size;
			// Get memory type index for a host visible buffer
			memAllocInfo.memoryTypeIndex = device->getMemoryType(memReqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

			stagingMemory = device->logicalDevice.allocateMemory(memAllocInfo);
			device->logicalDevice.bindBufferMemory(stagingBuffer, stagingMemory, 0);

			// Copy texture data into staging buffer
			uint8_t *data = (uint8_t*)device->logicalDevice.mapMemory(stagingMemory, 0, memReqs.size, vk::MemoryMapFlags());
			memcpy(data, buffer, bufferSize);
			device->logicalDevice.unmapMemory(stagingMemory);

			vk::BufferImageCopy bufferCopyRegion = {};
			bufferCopyRegion.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
			bufferCopyRegion.imageSubresource.mipLevel = 0;
			bufferCopyRegion.imageSubresource.baseArrayLayer = 0;
			bufferCopyRegion.imageSubresource.layerCount = 1;
			bufferCopyRegion.imageExtent.width = width;
			bufferCopyRegion.imageExtent.height = height;
			bufferCopyRegion.imageExtent.depth = 1;
			bufferCopyRegion.bufferOffset = 0;

			// Create optimal tiled target image
			vk::ImageCreateInfo imageCreateInfo = vks::initializers::imageCreateInfo();
			imageCreateInfo.imageType = vk::ImageType::e2D;
			imageCreateInfo.format = format;
			imageCreateInfo.mipLevels = mipLevels;
			imageCreateInfo.arrayLayers = 1;
			imageCreateInfo.samples = vk::SampleCountFlagBits::e1;
			imageCreateInfo.tiling = vk::ImageTiling::eOptimal;
			imageCreateInfo.sharingMode = vk::SharingMode::eExclusive;
			imageCreateInfo.initialLayout = vk::ImageLayout::eUndefined;
			imageCreateInfo.extent = vk::Extent3D{ width, height, 1 };
			imageCreateInfo.usage = imageUsageFlags;
			// Ensure that the TRANSFER_DST bit is set for staging
			if (!(imageCreateInfo.usage & vk::ImageUsageFlagBits::eTransferDst))
			{
				imageCreateInfo.usage |= vk::ImageUsageFlagBits::eTransferDst;
			}
			image = device->logicalDevice.createImage(imageCreateInfo);

			memReqs = device->logicalDevice.getImageMemoryRequirements(image);

			memAllocInfo.allocationSize = memReqs.size;

			memAllocInfo.memoryTypeIndex = device->getMemoryType(memReqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);
			deviceMemory = device->logicalDevice.allocateMemory(memAllocInfo);
			device->logicalDevice.bindImageMemory(image, deviceMemory, 0);

			vk::ImageSubresourceRange subresourceRange;
			subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
			subresourceRange.baseMipLevel = 0;
			subresourceRange.levelCount = mipLevels;
			subresourceRange.layerCount = 1;

			// Image barrier for optimal image (target)
			// Optimal image will be used as destination for the copy
			vks::tools::setImageLayout(
				copyCmd,
				image,
				vk::ImageLayout::eUndefined,
				vk::ImageLayout::eTransferDstOptimal,
				subresourceRange);

			// Copy mip levels from staging buffer
			copyCmd.copyBufferToImage(
				stagingBuffer,
				image,
				vk::ImageLayout::eTransferDstOptimal,
				bufferCopyRegion
			);

			// Change texture image layout to shader read after all mip levels have been copied
			this->imageLayout = imageLayout;
			vks::tools::setImageLayout(
				copyCmd,
				image,
				vk::ImageLayout::eTransferDstOptimal,
				imageLayout,
				subresourceRange);

			device->flushCommandBuffer(copyCmd, copyQueue);

			// Clean up staging resources
			device->logicalDevice.freeMemory(stagingMemory);
			device->logicalDevice.destroyBuffer(stagingBuffer);

			// Create sampler
			vk::SamplerCreateInfo samplerCreateInfo = {};

			samplerCreateInfo.magFilter = filter;
			samplerCreateInfo.minFilter = filter;
			samplerCreateInfo.mipmapMode = vk::SamplerMipmapMode::eLinear;
			samplerCreateInfo.addressModeU = vk::SamplerAddressMode::eRepeat;
			samplerCreateInfo.addressModeV = vk::SamplerAddressMode::eRepeat;
			samplerCreateInfo.addressModeW = vk::SamplerAddressMode::eRepeat;
			samplerCreateInfo.mipLodBias = 0.0f;
			samplerCreateInfo.compareOp = vk::CompareOp::eNever;
			samplerCreateInfo.minLod = 0.0f;
			samplerCreateInfo.maxLod = 0.0f;
			samplerCreateInfo.maxAnisotropy = 1.0f;
			sampler = device->logicalDevice.createSampler(samplerCreateInfo);

			// Create image view
			vk::ImageViewCreateInfo viewCreateInfo = {};
			viewCreateInfo.viewType = vk::ImageViewType::e2D;
			viewCreateInfo.format = format;
			viewCreateInfo.components = { vk::ComponentSwizzle::eR, vk::ComponentSwizzle::eG, vk::ComponentSwizzle::eB, vk::ComponentSwizzle::eA };
			viewCreateInfo.subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 };
			viewCreateInfo.subresourceRange.levelCount = 1;
			viewCreateInfo.image = image;
			view = device->logicalDevice.createImageView(viewCreateInfo);

			// Update descriptor image info member that can be used for setting up descriptor sets
			updateDescriptor();
		}

	};

	/** @brief 2D array texture */
	class Texture2DArray : public Texture {
	public:
		/**
		* Load a 2D texture array including all mip levels
		*
		* @param filename File to load (supports .ktx and .dds)
		* @param format Vulkan format of the image data stored in the file
		* @param device Vulkan device to create the texture on
		* @param copyQueue Queue used for the texture staging copy commands (must support transfer)
		* @param (Optional) imageUsageFlags Usage flags for the texture's image (defaults to vk::ImageUsageFlagBits::eSampled)
		* @param (Optional) imageLayout Usage layout for the texture (defaults vk::ImageLayout::eShaderReadOnlyOptimal)
		*
		*/
		void loadFromFile(
			std::string filename,
			vk::Format format,
			vks::VulkanDevice *device,
			vk::Queue copyQueue,
			vk::ImageUsageFlags imageUsageFlags = vk::ImageUsageFlagBits::eSampled,
			vk::ImageLayout imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal)
		{
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

			gli::texture2d_array tex2DArray(gli::load((const char*)textureData, size));

			free(textureData);
#else
			if (!vks::tools::fileExists(filename)) {
				vks::tools::exitFatal("Could not load texture from " + filename, "File not found");
			}
			gli::texture2d_array tex2DArray(gli::load(filename));
#endif	
			assert(!tex2DArray.empty());

			this->device = device;
			width = static_cast<uint32_t>(tex2DArray.extent().x);
			height = static_cast<uint32_t>(tex2DArray.extent().y);
			layerCount = static_cast<uint32_t>(tex2DArray.layers());
			mipLevels = static_cast<uint32_t>(tex2DArray.levels());

			vk::MemoryAllocateInfo memAllocInfo = vks::initializers::memoryAllocateInfo();
			vk::MemoryRequirements memReqs;

			// Create a host-visible staging buffer that contains the raw image data
			vk::Buffer stagingBuffer;
			vk::DeviceMemory stagingMemory;

			vk::BufferCreateInfo bufferCreateInfo = vks::initializers::bufferCreateInfo();
			bufferCreateInfo.size = tex2DArray.size();
			// This buffer is used as a transfer source for the buffer copy
			bufferCreateInfo.usage = vk::BufferUsageFlagBits::eTransferSrc;
			bufferCreateInfo.sharingMode = vk::SharingMode::eExclusive;

			stagingBuffer = device->logicalDevice.createBuffer(bufferCreateInfo);

			// Get memory requirements for the staging buffer (alignment, memory type bits)
			memReqs = device->logicalDevice.getBufferMemoryRequirements(stagingBuffer);

			memAllocInfo.allocationSize = memReqs.size;
			// Get memory type index for a host visible buffer
			memAllocInfo.memoryTypeIndex = device->getMemoryType(memReqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

			stagingMemory = device->logicalDevice.allocateMemory(memAllocInfo);
			device->logicalDevice.bindBufferMemory(stagingBuffer, stagingMemory, 0);

			// Copy texture data into staging buffer
			uint8_t *data = (uint8_t*)device->logicalDevice.mapMemory(stagingMemory, 0, memReqs.size, vk::MemoryMapFlags());
			memcpy(data, tex2DArray.data(), static_cast<size_t>(tex2DArray.size()));
			device->logicalDevice.unmapMemory(stagingMemory);

			// Setup buffer copy regions for each layer including all of it's miplevels
			std::vector<vk::BufferImageCopy> bufferCopyRegions;
			size_t offset = 0;

			for (uint32_t layer = 0; layer < layerCount; layer++)
			{
				for (uint32_t level = 0; level < mipLevels; level++)
				{
					vk::BufferImageCopy bufferCopyRegion = {};
					bufferCopyRegion.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
					bufferCopyRegion.imageSubresource.mipLevel = level;
					bufferCopyRegion.imageSubresource.baseArrayLayer = layer;
					bufferCopyRegion.imageSubresource.layerCount = 1;
					bufferCopyRegion.imageExtent.width = static_cast<uint32_t>(tex2DArray[layer][level].extent().x);
					bufferCopyRegion.imageExtent.height = static_cast<uint32_t>(tex2DArray[layer][level].extent().y);
					bufferCopyRegion.imageExtent.depth = 1;
					bufferCopyRegion.bufferOffset = offset;

					bufferCopyRegions.push_back(bufferCopyRegion);

					// Increase offset into staging buffer for next level / face
					offset += tex2DArray[layer][level].size();
				}
			}

			// Create optimal tiled target image
			vk::ImageCreateInfo imageCreateInfo = vks::initializers::imageCreateInfo();
			imageCreateInfo.imageType = vk::ImageType::e2D;
			imageCreateInfo.format = format;
			imageCreateInfo.samples = vk::SampleCountFlagBits::e1;
			imageCreateInfo.tiling = vk::ImageTiling::eOptimal;
			imageCreateInfo.sharingMode = vk::SharingMode::eExclusive;
			imageCreateInfo.initialLayout = vk::ImageLayout::eUndefined;
			imageCreateInfo.extent = vk::Extent3D{ width, height, 1 };
			imageCreateInfo.usage = imageUsageFlags;
			// Ensure that the TRANSFER_DST bit is set for staging
			if (!(imageCreateInfo.usage & vk::ImageUsageFlagBits::eTransferDst))
			{
				imageCreateInfo.usage |= vk::ImageUsageFlagBits::eTransferDst;
			}
			imageCreateInfo.arrayLayers = layerCount;
			imageCreateInfo.mipLevels = mipLevels;

			image = device->logicalDevice.createImage(imageCreateInfo);

			memReqs = device->logicalDevice.getImageMemoryRequirements(image);

			memAllocInfo.allocationSize = memReqs.size;
			memAllocInfo.memoryTypeIndex = device->getMemoryType(memReqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);

			deviceMemory = device->logicalDevice.allocateMemory(memAllocInfo);
			device->logicalDevice.bindImageMemory(image, deviceMemory, 0);

			// Use a separate command buffer for texture loading
			vk::CommandBuffer copyCmd = device->createCommandBuffer(vk::CommandBufferLevel::ePrimary, true);

			// Image barrier for optimal image (target)
			// Set initial layout for all array layers (faces) of the optimal (target) tiled texture
			vk::ImageSubresourceRange subresourceRange;
			subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
			subresourceRange.baseMipLevel = 0;
			subresourceRange.levelCount = mipLevels;
			subresourceRange.layerCount = layerCount;

			vks::tools::setImageLayout(
				copyCmd,
				image,
				vk::ImageLayout::eUndefined,
				vk::ImageLayout::eTransferDstOptimal,
				subresourceRange);

			// Copy the layers and mip levels from the staging buffer to the optimal tiled image
			copyCmd.copyBufferToImage(
				stagingBuffer,
				image,
				vk::ImageLayout::eTransferDstOptimal,
				bufferCopyRegions);

			// Change texture image layout to shader read after all faces have been copied
			this->imageLayout = imageLayout;
			vks::tools::setImageLayout(
				copyCmd,
				image,
				vk::ImageLayout::eTransferDstOptimal,
				imageLayout,
				subresourceRange);

			device->flushCommandBuffer(copyCmd, copyQueue);

			// Create sampler
			vk::SamplerCreateInfo samplerCreateInfo = vks::initializers::samplerCreateInfo();
			samplerCreateInfo.magFilter = vk::Filter::eLinear;
			samplerCreateInfo.minFilter = vk::Filter::eLinear;
			samplerCreateInfo.mipmapMode = vk::SamplerMipmapMode::eLinear;
			samplerCreateInfo.addressModeU = vk::SamplerAddressMode::eClampToEdge;
			samplerCreateInfo.addressModeV = samplerCreateInfo.addressModeU;
			samplerCreateInfo.addressModeW = samplerCreateInfo.addressModeU;
			samplerCreateInfo.mipLodBias = 0.0f;
			samplerCreateInfo.maxAnisotropy = device->enabledFeatures.samplerAnisotropy ? device->properties.limits.maxSamplerAnisotropy : 1.0f;
			samplerCreateInfo.compareOp = vk::CompareOp::eNever;
			samplerCreateInfo.minLod = 0.0f;
			samplerCreateInfo.maxLod = (float)mipLevels;
			samplerCreateInfo.borderColor = vk::BorderColor::eFloatOpaqueWhite;
			sampler = device->logicalDevice.createSampler(samplerCreateInfo);

			// Create image view
			vk::ImageViewCreateInfo viewCreateInfo = vks::initializers::imageViewCreateInfo();
			viewCreateInfo.viewType = vk::ImageViewType::e2DArray;
			viewCreateInfo.format = format;
			viewCreateInfo.components = { vk::ComponentSwizzle::eR, vk::ComponentSwizzle::eG, vk::ComponentSwizzle::eB, vk::ComponentSwizzle::eA };
			viewCreateInfo.subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 };
			viewCreateInfo.subresourceRange.layerCount = layerCount;
			viewCreateInfo.subresourceRange.levelCount = mipLevels;
			viewCreateInfo.image = image;
			view = device->logicalDevice.createImageView(viewCreateInfo);

			// Clean up staging resources
			device->logicalDevice.freeMemory(stagingMemory);
			device->logicalDevice.destroyBuffer(stagingBuffer);

			// Update descriptor image info member that can be used for setting up descriptor sets
			updateDescriptor();
		}
	};

	/** @brief Cube map texture */
	class TextureCubeMap : public Texture {
	public:
		/**
		* Load a cubemap texture including all mip levels from a single file
		*
		* @param filename File to load (supports .ktx and .dds)
		* @param format Vulkan format of the image data stored in the file
		* @param device Vulkan device to create the texture on
		* @param copyQueue Queue used for the texture staging copy commands (must support transfer)
		* @param (Optional) imageUsageFlags Usage flags for the texture's image (defaults to vk::ImageUsageFlagBits::eSampled)
		* @param (Optional) imageLayout Usage layout for the texture (defaults vk::ImageLayout::eShaderReadOnlyOptimal)
		*
		*/
		void loadFromFile(
			std::string filename,
			vk::Format format,
			vks::VulkanDevice *device,
			vk::Queue copyQueue,
			vk::ImageUsageFlags imageUsageFlags = vk::ImageUsageFlagBits::eSampled,
			vk::ImageLayout imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal)
		{
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

			gli::texture_cube texCube(gli::load((const char*)textureData, size));

			free(textureData);
#else
			if (!vks::tools::fileExists(filename)) {
				vks::tools::exitFatal("Could not load texture from " + filename, "File not found");
			}
			gli::texture_cube texCube(gli::load(filename));
#endif	
			assert(!texCube.empty());

			this->device = device;
			width = static_cast<uint32_t>(texCube.extent().x);
			height = static_cast<uint32_t>(texCube.extent().y);
			mipLevels = static_cast<uint32_t>(texCube.levels());

			vk::MemoryAllocateInfo memAllocInfo = vks::initializers::memoryAllocateInfo();
			vk::MemoryRequirements memReqs;

			// Create a host-visible staging buffer that contains the raw image data
			vk::Buffer stagingBuffer;
			vk::DeviceMemory stagingMemory;

			vk::BufferCreateInfo bufferCreateInfo = vks::initializers::bufferCreateInfo();
			bufferCreateInfo.size = texCube.size();
			// This buffer is used as a transfer source for the buffer copy
			bufferCreateInfo.usage = vk::BufferUsageFlagBits::eTransferSrc;
			bufferCreateInfo.sharingMode = vk::SharingMode::eExclusive;

			stagingBuffer = device->logicalDevice.createBuffer(bufferCreateInfo);

			// Get memory requirements for the staging buffer (alignment, memory type bits)
			memReqs = device->logicalDevice.getBufferMemoryRequirements(stagingBuffer);

			memAllocInfo.allocationSize = memReqs.size;
			// Get memory type index for a host visible buffer
			memAllocInfo.memoryTypeIndex = device->getMemoryType(memReqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

			stagingMemory = device->logicalDevice.allocateMemory(memAllocInfo);
			device->logicalDevice.bindBufferMemory(stagingBuffer, stagingMemory, 0);

			// Copy texture data into staging buffer
			uint8_t *data = (uint8_t*)device->logicalDevice.mapMemory(stagingMemory, 0, memReqs.size, vk::MemoryMapFlags());
			memcpy(data, texCube.data(), texCube.size());
			device->logicalDevice.unmapMemory(stagingMemory);

			// Setup buffer copy regions for each face including all of it's miplevels
			std::vector<vk::BufferImageCopy> bufferCopyRegions;
			size_t offset = 0;

			for (uint32_t face = 0; face < 6; face++)
			{
				for (uint32_t level = 0; level < mipLevels; level++)
				{
					vk::BufferImageCopy bufferCopyRegion = {};
					bufferCopyRegion.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
					bufferCopyRegion.imageSubresource.mipLevel = level;
					bufferCopyRegion.imageSubresource.baseArrayLayer = face;
					bufferCopyRegion.imageSubresource.layerCount = 1;
					bufferCopyRegion.imageExtent.width = static_cast<uint32_t>(texCube[face][level].extent().x);
					bufferCopyRegion.imageExtent.height = static_cast<uint32_t>(texCube[face][level].extent().y);
					bufferCopyRegion.imageExtent.depth = 1;
					bufferCopyRegion.bufferOffset = offset;

					bufferCopyRegions.push_back(bufferCopyRegion);

					// Increase offset into staging buffer for next level / face
					offset += texCube[face][level].size();
				}
			}

			// Create optimal tiled target image
			vk::ImageCreateInfo imageCreateInfo = vks::initializers::imageCreateInfo();
			imageCreateInfo.imageType = vk::ImageType::e2D;
			imageCreateInfo.format = format;
			imageCreateInfo.mipLevels = mipLevels;
			imageCreateInfo.samples = vk::SampleCountFlagBits::e1;
			imageCreateInfo.tiling = vk::ImageTiling::eOptimal;
			imageCreateInfo.sharingMode = vk::SharingMode::eExclusive;
			imageCreateInfo.initialLayout = vk::ImageLayout::eUndefined;
			imageCreateInfo.extent = vk::Extent3D{ width, height, 1 };
			imageCreateInfo.usage = imageUsageFlags;
			// Ensure that the TRANSFER_DST bit is set for staging
			if (~(imageCreateInfo.usage & vk::ImageUsageFlagBits::eTransferDst))
			{
				imageCreateInfo.usage |= vk::ImageUsageFlagBits::eTransferDst;
			}
			// Cube faces count as array layers in Vulkan
			imageCreateInfo.arrayLayers = 6;
			// This flag is required for cube map images
			imageCreateInfo.flags = vk::ImageCreateFlagBits::eCubeCompatible;


			image = device->logicalDevice.createImage(imageCreateInfo);

			memReqs = device->logicalDevice.getImageMemoryRequirements(image);

			memAllocInfo.allocationSize = memReqs.size;
			memAllocInfo.memoryTypeIndex = device->getMemoryType(memReqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);

			deviceMemory = device->logicalDevice.allocateMemory(memAllocInfo);
			device->logicalDevice.bindImageMemory(image, deviceMemory, 0);

			// Use a separate command buffer for texture loading
			vk::CommandBuffer copyCmd = device->createCommandBuffer(vk::CommandBufferLevel::ePrimary, true);

			// Image barrier for optimal image (target)
			// Set initial layout for all array layers (faces) of the optimal (target) tiled texture
			vk::ImageSubresourceRange subresourceRange;
			subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
			subresourceRange.baseMipLevel = 0;
			subresourceRange.levelCount = mipLevels;
			subresourceRange.layerCount = 6;

			vks::tools::setImageLayout(
				copyCmd,
				image,
				vk::ImageLayout::eUndefined,
				vk::ImageLayout::eTransferDstOptimal,
				subresourceRange);

			// Copy the cube map faces from the staging buffer to the optimal tiled image
			copyCmd.copyBufferToImage(
				stagingBuffer,
				image,
				vk::ImageLayout::eTransferDstOptimal,
				bufferCopyRegions);

			// Change texture image layout to shader read after all faces have been copied
			this->imageLayout = imageLayout;
			vks::tools::setImageLayout(
				copyCmd,
				image,
				vk::ImageLayout::eTransferDstOptimal,
				imageLayout,
				subresourceRange);

			device->flushCommandBuffer(copyCmd, copyQueue);

			// Create sampler
			vk::SamplerCreateInfo samplerCreateInfo = vks::initializers::samplerCreateInfo();
			samplerCreateInfo.magFilter = vk::Filter::eLinear;
			samplerCreateInfo.minFilter = vk::Filter::eLinear;
			samplerCreateInfo.mipmapMode = vk::SamplerMipmapMode::eLinear;
			samplerCreateInfo.addressModeU = vk::SamplerAddressMode::eClampToEdge;
			samplerCreateInfo.addressModeV = samplerCreateInfo.addressModeU;
			samplerCreateInfo.addressModeW = samplerCreateInfo.addressModeU;
			samplerCreateInfo.mipLodBias = 0.0f;
			samplerCreateInfo.maxAnisotropy = device->enabledFeatures.samplerAnisotropy ? device->properties.limits.maxSamplerAnisotropy : 1.0f;
			samplerCreateInfo.compareOp = vk::CompareOp::eNever;
			samplerCreateInfo.minLod = 0.0f;
			samplerCreateInfo.maxLod = (float)mipLevels;
			samplerCreateInfo.borderColor = vk::BorderColor::eFloatOpaqueWhite;
			sampler = device->logicalDevice.createSampler(samplerCreateInfo);

			// Create image view
			vk::ImageViewCreateInfo viewCreateInfo = vks::initializers::imageViewCreateInfo();
			viewCreateInfo.viewType = vk::ImageViewType::eCube;
			viewCreateInfo.format = format;
			viewCreateInfo.components = { vk::ComponentSwizzle::eR, vk::ComponentSwizzle::eG, vk::ComponentSwizzle::eB, vk::ComponentSwizzle::eA };
			viewCreateInfo.subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 };
			viewCreateInfo.subresourceRange.layerCount = 6;
			viewCreateInfo.subresourceRange.levelCount = mipLevels;
			viewCreateInfo.image = image;
			view = device->logicalDevice.createImageView(viewCreateInfo);

			// Clean up staging resources
			device->logicalDevice.freeMemory(stagingMemory);
			device->logicalDevice.destroyBuffer(stagingBuffer);

			// Update descriptor image info member that can be used for setting up descriptor sets
			updateDescriptor();
		}
	};

}