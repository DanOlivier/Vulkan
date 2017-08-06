/*
* Assorted commonly used Vulkan helper functions
*
* Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include "VulkanTools.h"

namespace vks
{
	namespace tools
	{
		vk::Bool32 getSupportedDepthFormat(vk::PhysicalDevice physicalDevice, vk::Format *depthFormat)
		{
			// Since all depth formats may be optional, we need to find a suitable depth format to use
			// Start with the highest precision packed format
			std::vector<vk::Format> depthFormats = {
				vk::Format::eD32SfloatS8Uint,
				vk::Format::eD32Sfloat,
				vk::Format::eD24UnormS8Uint,
				vk::Format::eD16UnormS8Uint,
				vk::Format::eD16Unorm
			};

			for (auto& format : depthFormats)
			{
				vk::FormatProperties formatProps;
				formatProps = physicalDevice.getFormatProperties(format);
				// Format must support depth stencil attachment for optimal tiling
				if (formatProps.optimalTilingFeatures & vk::FormatFeatureFlagBits::eDepthStencilAttachment)
				{
					*depthFormat = format;
					return true;
				}
			}

			return false;
		}

		// Create an image memory barrier for changing the layout of
		// an image and put it into an active command buffer
		// See chapter 11.4 "Image Layout" for details

		void setImageLayout(
			vk::CommandBuffer cmdbuffer,
			vk::Image image,
			vk::ImageLayout oldImageLayout,
			vk::ImageLayout newImageLayout,
			vk::ImageSubresourceRange subresourceRange,
			vk::PipelineStageFlags srcStageMask,
			vk::PipelineStageFlags dstStageMask)
		{
			// Create an image barrier object
			vk::ImageMemoryBarrier imageMemoryBarrier = vks::initializers::imageMemoryBarrier();
			imageMemoryBarrier.oldLayout = oldImageLayout;
			imageMemoryBarrier.newLayout = newImageLayout;
			imageMemoryBarrier.image = image;
			imageMemoryBarrier.subresourceRange = subresourceRange;

			// Source layouts (old)
			// Source access mask controls actions that have to be finished on the old layout
			// before it will be transitioned to the new layout
			switch (oldImageLayout)
			{
			case vk::ImageLayout::eUndefined:
				// Image layout is undefined (or does not matter)
				// Only valid as initial layout
				// No flags required, listed only for completeness
				imageMemoryBarrier.srcAccessMask = vk::AccessFlagBits();
				break;

			case vk::ImageLayout::ePreinitialized:
				// Image is preinitialized
				// Only valid as initial layout for linear images, preserves memory contents
				// Make sure host writes have been finished
				imageMemoryBarrier.srcAccessMask = vk::AccessFlagBits::eHostWrite;
				break;

			case vk::ImageLayout::eColorAttachmentOptimal:
				// Image is a color attachment
				// Make sure any writes to the color buffer have been finished
				imageMemoryBarrier.srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite;
				break;

			case vk::ImageLayout::eDepthStencilAttachmentOptimal:
				// Image is a depth/stencil attachment
				// Make sure any writes to the depth/stencil buffer have been finished
				imageMemoryBarrier.srcAccessMask = vk::AccessFlagBits::eDepthStencilAttachmentWrite;
				break;

			case vk::ImageLayout::eTransferSrcOptimal:
				// Image is a transfer source 
				// Make sure any reads from the image have been finished
				imageMemoryBarrier.srcAccessMask = vk::AccessFlagBits::eTransferRead;
				break;

			case vk::ImageLayout::eTransferDstOptimal:
				// Image is a transfer destination
				// Make sure any writes to the image have been finished
				imageMemoryBarrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
				break;

			case vk::ImageLayout::eShaderReadOnlyOptimal:
				// Image is read by a shader
				// Make sure any shader reads from the image have been finished
				imageMemoryBarrier.srcAccessMask = vk::AccessFlagBits::eShaderRead;
				break;
			default:
				// Other source layouts aren't handled (yet)
				break;
			}

			// Target layouts (new)
			// Destination access mask controls the dependency for the new image layout
			switch (newImageLayout)
			{
			case vk::ImageLayout::eTransferDstOptimal:
				// Image will be used as a transfer destination
				// Make sure any writes to the image have been finished
				imageMemoryBarrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;
				break;

			case vk::ImageLayout::eTransferSrcOptimal:
				// Image will be used as a transfer source
				// Make sure any reads from the image have been finished
				imageMemoryBarrier.dstAccessMask = vk::AccessFlagBits::eTransferRead;
				break;

			case vk::ImageLayout::eColorAttachmentOptimal:
				// Image will be used as a color attachment
				// Make sure any writes to the color buffer have been finished
				imageMemoryBarrier.dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite;
				break;

			case vk::ImageLayout::eDepthStencilAttachmentOptimal:
				// Image layout will be used as a depth/stencil attachment
				// Make sure any writes to depth/stencil buffer have been finished
				imageMemoryBarrier.dstAccessMask = imageMemoryBarrier.dstAccessMask | vk::AccessFlagBits::eDepthStencilAttachmentWrite;
				break;

			case vk::ImageLayout::eShaderReadOnlyOptimal:
				// Image will be read in a shader (sampler, input attachment)
				// Make sure any writes to the image have been finished
				if (!imageMemoryBarrier.srcAccessMask)
				{
					imageMemoryBarrier.srcAccessMask = vk::AccessFlagBits::eHostWrite | vk::AccessFlagBits::eTransferWrite;
				}
				imageMemoryBarrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;
				break;
			default:
				// Other source layouts aren't handled (yet)
				break;
			}

			// Put barrier inside setup command buffer
			cmdbuffer.pipelineBarrier(
				srcStageMask,
				dstStageMask,
				vk::DependencyFlags(),
				nullptr,
				nullptr,
				imageMemoryBarrier);
		}

		// Fixed sub resource on first mip level and layer
		void setImageLayout(
			vk::CommandBuffer cmdbuffer,
			vk::Image image,
			vk::ImageAspectFlags aspectMask,
			vk::ImageLayout oldImageLayout,
			vk::ImageLayout newImageLayout,
			vk::PipelineStageFlags srcStageMask,
			vk::PipelineStageFlags dstStageMask)
		{
			vk::ImageSubresourceRange subresourceRange;
			subresourceRange.aspectMask = aspectMask;
			subresourceRange.baseMipLevel = 0;
			subresourceRange.levelCount = 1;
			subresourceRange.layerCount = 1;
			setImageLayout(cmdbuffer, image, oldImageLayout, newImageLayout, subresourceRange, srcStageMask, dstStageMask);
		}

		void insertImageMemoryBarrier(
			vk::CommandBuffer cmdbuffer,
			vk::Image image,
			vk::AccessFlags srcAccessMask,
			vk::AccessFlags dstAccessMask,
			vk::ImageLayout oldImageLayout,
			vk::ImageLayout newImageLayout,
			vk::PipelineStageFlags srcStageMask,
			vk::PipelineStageFlags dstStageMask,
			vk::ImageSubresourceRange subresourceRange)
		{
			vk::ImageMemoryBarrier imageMemoryBarrier = vks::initializers::imageMemoryBarrier();
			imageMemoryBarrier.srcAccessMask = srcAccessMask;
			imageMemoryBarrier.dstAccessMask = dstAccessMask;
			imageMemoryBarrier.oldLayout = oldImageLayout;
			imageMemoryBarrier.newLayout = newImageLayout;
			imageMemoryBarrier.image = image;
			imageMemoryBarrier.subresourceRange = subresourceRange;

			cmdbuffer.pipelineBarrier(
				srcStageMask,
				dstStageMask,
				vk::DependencyFlags(),
				nullptr,
				nullptr,
				imageMemoryBarrier);
		}

		void exitFatal(std::string message, std::string caption)
		{
#if defined(_WIN32)
			MessageBox(NULL, message.c_str(), caption.c_str(), MB_OK | MB_ICONERROR);
#elif defined(__ANDROID__)	
			LOGE("Fatal error: %s", message.c_str());
#else
			std::cerr << message << "\n";
#endif
			exit(1);
		}

		std::string readTextFile(const char *fileName)
		{
			std::string fileContent;
			std::ifstream fileStream(fileName, std::ios::in);
			if (!fileStream.is_open()) {
				printf("File %s not found\n", fileName);
				return "";
			}
			std::string line = "";
			while (!fileStream.eof()) {
				getline(fileStream, line);
				fileContent.append(line + "\n");
			}
			fileStream.close();
			return fileContent;
		}

#if defined(__ANDROID__)
		// Android shaders are stored as assets in the apk
		// So they need to be loaded via the asset manager
		vk::ShaderModule loadShader(AAssetManager* assetManager, const char *fileName, vk::Device device)
		{
			// Load shader from compressed asset
			AAsset* asset = AAssetManager_open(assetManager, fileName, AASSET_MODE_STREAMING);
			assert(asset);
			size_t size = AAsset_getLength(asset);
			assert(size > 0);

			char *shaderCode = new char[size];
			AAsset_read(asset, shaderCode, size);
			AAsset_close(asset);

			vk::ShaderModule shaderModule;
			vk::ShaderModuleCreateInfo moduleCreateInfo;
			moduleCreateInfo.codeSize = size;
			moduleCreateInfo.pCode = (uint32_t*)shaderCode;
			moduleCreateInfo.flags = 0;

			shaderModule = device.createShaderModule(moduleCreateInfo);

			delete[] shaderCode;

			return shaderModule;
		}
#else
		vk::ShaderModule loadShader(const char *fileName, vk::Device device)
		{
			std::ifstream is(fileName, std::ios::binary | std::ios::in | std::ios::ate);

			if (is.is_open())
			{
				size_t size = is.tellg();
				is.seekg(0, std::ios::beg);
				char* shaderCode = new char[size];
				is.read(shaderCode, size);
				is.close();

				assert(size > 0);

				vk::ShaderModule shaderModule;
				vk::ShaderModuleCreateInfo moduleCreateInfo{};

				moduleCreateInfo.codeSize = size;
				moduleCreateInfo.pCode = (uint32_t*)shaderCode;

				shaderModule = device.createShaderModule(moduleCreateInfo);

				delete[] shaderCode;

				return shaderModule;
			}
			else
			{
				std::cerr << "Error: Could not open shader file \"" << fileName << "\"" << std::endl;
				return nullptr;
			}
		}
#endif

		vk::ShaderModule loadShaderGLSL(const char *fileName, vk::Device device, vk::ShaderStageFlagBits stage)
		{
			std::string shaderSrc = readTextFile(fileName);
			const char *shaderCode = shaderSrc.c_str();
			size_t size = strlen(shaderCode);
			assert(size > 0);

			vk::ShaderModule shaderModule;
			vk::ShaderModuleCreateInfo moduleCreateInfo;
			moduleCreateInfo.codeSize = 3 * sizeof(uint32_t) + size + 1;
			moduleCreateInfo.pCode = (uint32_t*)malloc(moduleCreateInfo.codeSize);
			//moduleCreateInfo.flags = 0;

			// Magic SPV number
			((uint32_t *)moduleCreateInfo.pCode)[0] = 0x07230203;
			((uint32_t *)moduleCreateInfo.pCode)[1] = 0;
			((uint32_t *)moduleCreateInfo.pCode)[2] = (uint32_t)stage;
			memcpy(((uint32_t *)moduleCreateInfo.pCode + 3), shaderCode, size + 1);

			shaderModule = device.createShaderModule(moduleCreateInfo);

			return shaderModule;
		}

		bool fileExists(const std::string &filename)
		{
			std::ifstream f(filename.c_str());
			return !f.fail();
		}
	}
}