/*
* Assorted Vulkan helper functions
*
* Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#pragma once

#include "vulkan/vulkan.hpp"

#include <assert.h>
#include <iostream>
#if defined(_WIN32)
#elif defined(__ANDROID__)
#endif

// Custom define for better code readability
#define VK_FLAGS_NONE 0

// Macro to check and display Vulkan return results
#if defined(__ANDROID__)
#define VK_CHECK_RESULT(f)																				\
{																										\
	vk::Result res = (f);																					\
	if (res != vk::Result::eSuccess)																				\
	{																									\
		LOGE("Fatal : vk::Result is \" %s \" in %s at line %d", vk::to_string(res).c_str(), __FILE__, __LINE__); \
		assert(res == vk::Result::eSuccess);																		\
	}																									\
}
#else
#define VK_CHECK_RESULT(f)																				\
{																										\
	vk::Result res = (f);																					\
	if (res != vk::Result::eSuccess)																				\
	{																									\
		std::cout << "Fatal : vk::Result is \"" << vk::to_string(res) << "\" in " << __FILE__ << " at line " << __LINE__ << std::endl; \
		assert(res == vk::Result::eSuccess);																		\
	}																									\
}
#endif

#if defined(__ANDROID__)
#define ASSET_PATH ""
#else
#define ASSET_PATH "./../data/"
#endif

namespace vks
{
	namespace tools
	{
		// Selected a suitable supported depth format starting with 32 bit down to 16 bit
		// Returns false if none of the depth formats in the list is supported by the device
		vk::Bool32 getSupportedDepthFormat(vk::PhysicalDevice physicalDevice, vk::Format *depthFormat);

		// Put an image memory barrier for setting an image layout on the sub resource into the given command buffer
		void setImageLayout(
			vk::CommandBuffer cmdbuffer,
			vk::Image image,
			vk::ImageLayout oldImageLayout,
			vk::ImageLayout newImageLayout,
			vk::ImageSubresourceRange subresourceRange,
			vk::PipelineStageFlags srcStageMask = vk::PipelineStageFlagBits::eAllCommands,
			vk::PipelineStageFlags dstStageMask = vk::PipelineStageFlagBits::eAllCommands);
		// Uses a fixed sub resource layout with first mip level and layer
		void setImageLayout(
			vk::CommandBuffer cmdbuffer,
			vk::Image image,
			vk::ImageAspectFlags aspectMask,
			vk::ImageLayout oldImageLayout,
			vk::ImageLayout newImageLayout,
			vk::PipelineStageFlags srcStageMask = vk::PipelineStageFlagBits::eAllCommands,
			vk::PipelineStageFlags dstStageMask = vk::PipelineStageFlagBits::eAllCommands);

		/** @brief Inser an image memory barrier into the command buffer */
		void insertImageMemoryBarrier(
			vk::CommandBuffer cmdbuffer,
			vk::Image image,
			vk::AccessFlags srcAccessMask,
			vk::AccessFlags dstAccessMask,
			vk::ImageLayout oldImageLayout,
			vk::ImageLayout newImageLayout,
			vk::PipelineStageFlags srcStageMask,
			vk::PipelineStageFlags dstStageMask,
			vk::ImageSubresourceRange subresourceRange);

		// Display error message and exit on fatal error
		void exitFatal(std::string message, std::string caption);

		// Load a SPIR-V shader (binary) 
#if defined(__ANDROID__)
		vk::ShaderModule loadShader(AAssetManager* assetManager, const char *fileName, vk::Device device);
#else
		vk::ShaderModule loadShader(const char *fileName, vk::Device device);
#endif

		// Load a GLSL shader (text)
		// Note: GLSL support requires vendor-specific extensions to be enabled and is not a core-feature of Vulkan
		vk::ShaderModule loadShaderGLSL(const char *fileName, vk::Device device, vk::ShaderStageFlagBits stage);

		/** @brief Checks if a file exists */
		bool fileExists(const std::string &filename);
	}
}
