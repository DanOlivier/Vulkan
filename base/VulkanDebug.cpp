/*
* Vulkan examples debug wrapper
* 
* Appendix for VK_EXT_Debug_Report can be found at https://github.com/KhronosGroup/Vulkan-Docs/blob/1.0-VK_EXT_debug_report/doc/specs/vulkan/appendices/debug_report.txt
*
* Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include "VulkanDebug.h"
#include <iostream>

namespace vks
{
	namespace debug
	{
#if !defined(__ANDROID__)
		// On desktop the LunarG loaders exposes a meta layer that contains all layers
		int32_t validationLayerCount = 1;
		const char *validationLayerNames[] = {
			"VK_LAYER_LUNARG_standard_validation"
		};
#else
		// On Android we need to explicitly select all layers
		int32_t validationLayerCount = 6;
		const char *validationLayerNames[] = {
			"VK_LAYER_GOOGLE_threading",
			"VK_LAYER_LUNARG_parameter_validation",
			"VK_LAYER_LUNARG_object_tracker",
			"VK_LAYER_LUNARG_core_validation",
			"VK_LAYER_LUNARG_swapchain",
			"VK_LAYER_GOOGLE_unique_objects"
		};
#endif

		PFN_vkCreateDebugReportCallbackEXT CreateDebugReportCallback = nullptr;
		PFN_vkDestroyDebugReportCallbackEXT DestroyDebugReportCallback = nullptr;
		PFN_vkDebugReportMessageEXT dbgBreakCallback = nullptr;

		vk::DebugReportCallbackEXT msgCallback;

		vk::Bool32 messageCallback(
			vk::DebugReportFlagsEXT flags,
			vk::DebugReportObjectTypeEXT objType,
			uint64_t srcObject,
			size_t location,
			int32_t msgCode,
			const char* pLayerPrefix,
			const char* pMsg,
			void* pUserData)
		{
			// Select prefix depending on flags passed to the callback
			// Note that multiple flags may be set for a single validation message
			std::string prefix("");

			// Error that may result in undefined behaviour
			if (flags & vk::DebugReportFlagBitsEXT::eError)
			{
				prefix += "ERROR:";
			};
			// Warnings may hint at unexpected / non-spec API usage
			if (flags & vk::DebugReportFlagBitsEXT::eWarning)
			{
				prefix += "WARNING:";
			};
			// May indicate sub-optimal usage of the API
			if (flags & vk::DebugReportFlagBitsEXT::ePerformanceWarning)
			{
				prefix += "PERFORMANCE:";
			};
			// Informal messages that may become handy during debugging
			if (flags & vk::DebugReportFlagBitsEXT::eInformation)
			{
				prefix += "INFO:";
			}
			// Diagnostic info from the Vulkan loader and layers
			// Usually not helpful in terms of API usage, but may help to debug layer and loader problems 
			if (flags & vk::DebugReportFlagBitsEXT::eDebug)
			{
				prefix += "DEBUG:";
			}

			// Display message to default output (console/logcat)
			std::stringstream debugMessage;
			debugMessage << prefix << " [" << pLayerPrefix << "] Code " << msgCode << " : " << pMsg;

#if defined(__ANDROID__)
			if (flags & vk::DebugReportFlagBitsEXT::eError) {
				LOGE("%s", debugMessage.str().c_str());
			}
			else {
				LOGD("%s", debugMessage.str().c_str());
			}
#else
			if (flags & vk::DebugReportFlagBitsEXT::eError) {
				std::cerr << debugMessage.str() << "\n";
			}
			else {
				std::cout << debugMessage.str() << "\n";
			}
#endif

			fflush(stdout);

			// The return value of this callback controls wether the Vulkan call that caused
			// the validation message will be aborted or not
			// We return VK_FALSE as we DON'T want Vulkan calls that cause a validation message 
			// (and return a vk::Result) to abort
			// If you instead want to have calls abort, pass in VK_TRUE and the function will 
			// return vk::Result::eErrorValidationFailedEXT 
			return VK_FALSE;
		}

		void setupDebugging(vk::Instance instance, vk::DebugReportFlagsEXT flags, vk::DebugReportCallbackEXT callBack)
		{
			CreateDebugReportCallback = reinterpret_cast<PFN_vkCreateDebugReportCallbackEXT>(vkGetInstanceProcAddr(instance, "vkCreateDebugReportCallbackEXT"));
			DestroyDebugReportCallback = reinterpret_cast<PFN_vkDestroyDebugReportCallbackEXT>(vkGetInstanceProcAddr(instance, "vkDestroyDebugReportCallbackEXT"));
			dbgBreakCallback = reinterpret_cast<PFN_vkDebugReportMessageEXT>(vkGetInstanceProcAddr(instance, "vkDebugReportMessageEXT"));

			vk::DebugReportCallbackCreateInfoEXT dbgCreateInfo = {};

			dbgCreateInfo.pfnCallback = (PFN_vkDebugReportCallbackEXT)messageCallback;
			dbgCreateInfo.flags = flags;

			if(callBack)
				callBack = instance.createDebugReportCallbackEXT(dbgCreateInfo);
			else
				msgCallback = instance.createDebugReportCallbackEXT(dbgCreateInfo);
		}

		void freeDebugCallback(vk::Instance instance)
		{
			if (msgCallback)
			{
				instance.destroyDebugReportCallbackEXT(msgCallback);
			}
		}
	}

	namespace debugmarker
	{
		bool active = false;

		PFN_vkDebugMarkerSetObjectTagEXT pfnDebugMarkerSetObjectTag = nullptr;
		PFN_vkDebugMarkerSetObjectNameEXT pfnDebugMarkerSetObjectName = nullptr;
		PFN_vkCmdDebugMarkerBeginEXT pfnCmdDebugMarkerBegin = nullptr;
		PFN_vkCmdDebugMarkerEndEXT pfnCmdDebugMarkerEnd = nullptr;
		PFN_vkCmdDebugMarkerInsertEXT pfnCmdDebugMarkerInsert = nullptr;

		void setup(vk::Device device)
		{
			pfnDebugMarkerSetObjectTag = reinterpret_cast<PFN_vkDebugMarkerSetObjectTagEXT>(vkGetDeviceProcAddr(device, "vkDebugMarkerSetObjectTagEXT"));
			pfnDebugMarkerSetObjectName = reinterpret_cast<PFN_vkDebugMarkerSetObjectNameEXT>(vkGetDeviceProcAddr(device, "vkDebugMarkerSetObjectNameEXT"));
			pfnCmdDebugMarkerBegin = reinterpret_cast<PFN_vkCmdDebugMarkerBeginEXT>(vkGetDeviceProcAddr(device, "vkCmdDebugMarkerBeginEXT"));
			pfnCmdDebugMarkerEnd = reinterpret_cast<PFN_vkCmdDebugMarkerEndEXT>(vkGetDeviceProcAddr(device, "vkCmdDebugMarkerEndEXT"));
			pfnCmdDebugMarkerInsert = reinterpret_cast<PFN_vkCmdDebugMarkerInsertEXT>(vkGetDeviceProcAddr(device, "vkCmdDebugMarkerInsertEXT"));

			// Set flag if at least one function pointer is present
			active = (pfnDebugMarkerSetObjectName != nullptr);
		}

		void setObjectName(vk::Device device, uint64_t object, vk::DebugReportObjectTypeEXT objectType, const char *name)
		{
			// Check for valid function pointer (may not be present if not running in a debugging application)
			if (pfnDebugMarkerSetObjectName)
			{
				vk::DebugMarkerObjectNameInfoEXT nameInfo = {};

				nameInfo.objectType = objectType;
				nameInfo.object = object;
				nameInfo.pObjectName = name;
				pfnDebugMarkerSetObjectName(device, &(VkDebugMarkerObjectNameInfoEXT&)nameInfo);
			}
		}

		void setObjectTag(vk::Device device, uint64_t object, vk::DebugReportObjectTypeEXT objectType, uint64_t name, size_t tagSize, const void* tag)
		{
			// Check for valid function pointer (may not be present if not running in a debugging application)
			if (pfnDebugMarkerSetObjectTag)
			{
				vk::DebugMarkerObjectTagInfoEXT tagInfo = {};

				tagInfo.objectType = objectType;
				tagInfo.object = object;
				tagInfo.tagName = name;
				tagInfo.tagSize = tagSize;
				tagInfo.pTag = tag;
				pfnDebugMarkerSetObjectTag(device, &(VkDebugMarkerObjectTagInfoEXT&)tagInfo);
			}
		}

		void beginRegion(vk::CommandBuffer cmdbuffer, const char* pMarkerName, glm::vec4 color)
		{
			// Check for valid function pointer (may not be present if not running in a debugging application)
			if (pfnCmdDebugMarkerBegin)
			{
				vk::DebugMarkerMarkerInfoEXT markerInfo = {};

				memcpy(markerInfo.color, &color[0], sizeof(float) * 4);
				markerInfo.pMarkerName = pMarkerName;
				cmdbuffer.debugMarkerBeginEXT(&markerInfo);
			}
		}

		void insert(vk::CommandBuffer cmdbuffer, std::string markerName, glm::vec4 color)
		{
			// Check for valid function pointer (may not be present if not running in a debugging application)
			if (pfnCmdDebugMarkerInsert)
			{
				vk::DebugMarkerMarkerInfoEXT markerInfo = {};

				memcpy(markerInfo.color, &color[0], sizeof(float) * 4);
				markerInfo.pMarkerName = markerName.c_str();
				cmdbuffer.debugMarkerInsertEXT(&markerInfo);
			}
		}

		void endRegion(vk::CommandBuffer cmdBuffer)
		{
			// Check for valid function (may not be present if not runnin in a debugging application)
			if (pfnCmdDebugMarkerEnd)
			{
				cmdBuffer.debugMarkerEndEXT();
			}
		}

		void setCommandBufferName(vk::Device device, vk::CommandBuffer cmdBuffer, const char * name)
		{
			setObjectName(device, (uint64_t)(VkCommandBuffer)cmdBuffer, vk::DebugReportObjectTypeEXT::eCommandBuffer, name);
		}

		void setQueueName(vk::Device device, vk::Queue queue, const char * name)
		{
			setObjectName(device, (uint64_t)(VkQueue)queue, vk::DebugReportObjectTypeEXT::eQueue, name);
		}

		void setImageName(vk::Device device, vk::Image image, const char * name)
		{
			setObjectName(device, (uint64_t)(VkImage)image, vk::DebugReportObjectTypeEXT::eImage, name);
		}

		void setSamplerName(vk::Device device, vk::Sampler sampler, const char * name)
		{
			setObjectName(device, (uint64_t)(VkSampler)sampler, vk::DebugReportObjectTypeEXT::eSampler, name);
		}

		void setBufferName(vk::Device device, vk::Buffer buffer, const char * name)
		{
			setObjectName(device, (uint64_t)(VkBuffer)buffer, vk::DebugReportObjectTypeEXT::eBuffer, name);
		}

		void setDeviceMemoryName(vk::Device device, vk::DeviceMemory memory, const char * name)
		{
			setObjectName(device, (uint64_t)(VkDeviceMemory)memory, vk::DebugReportObjectTypeEXT::eDeviceMemory, name);
		}

		void setShaderModuleName(vk::Device device, vk::ShaderModule shaderModule, const char * name)
		{
			setObjectName(device, (uint64_t)(VkShaderModule)shaderModule, vk::DebugReportObjectTypeEXT::eShaderModule, name);
		}

		void setPipelineName(vk::Device device, vk::Pipeline pipeline, const char * name)
		{
			setObjectName(device, (uint64_t)(VkPipeline)pipeline, vk::DebugReportObjectTypeEXT::ePipeline, name);
		}

		void setPipelineLayoutName(vk::Device device, vk::PipelineLayout pipelineLayout, const char * name)
		{
			setObjectName(device, (uint64_t)(VkPipelineLayout)pipelineLayout, vk::DebugReportObjectTypeEXT::ePipelineLayout, name);
		}

		void setRenderPassName(vk::Device device, vk::RenderPass renderPass, const char * name)
		{
			setObjectName(device, (uint64_t)(VkRenderPass)renderPass, vk::DebugReportObjectTypeEXT::eRenderPass, name);
		}

		void setFramebufferName(vk::Device device, vk::Framebuffer framebuffer, const char * name)
		{
			setObjectName(device, (uint64_t)(VkFramebuffer)framebuffer, vk::DebugReportObjectTypeEXT::eFramebuffer, name);
		}

		void setDescriptorSetLayoutName(vk::Device device, vk::DescriptorSetLayout descriptorSetLayout, const char * name)
		{
			setObjectName(device, (uint64_t)(VkDescriptorSetLayout)descriptorSetLayout, vk::DebugReportObjectTypeEXT::eDescriptorSetLayout, name);
		}

		void setDescriptorSetName(vk::Device device, vk::DescriptorSet descriptorSet, const char * name)
		{
			setObjectName(device, (uint64_t)(VkDescriptorSet)descriptorSet, vk::DebugReportObjectTypeEXT::eDescriptorSet, name);
		}

		void setSemaphoreName(vk::Device device, vk::Semaphore semaphore, const char * name)
		{
			setObjectName(device, (uint64_t)(VkSemaphore)semaphore, vk::DebugReportObjectTypeEXT::eSemaphore, name);
		}

		void setFenceName(vk::Device device, vk::Fence fence, const char * name)
		{
			setObjectName(device, (uint64_t)(VkFence)fence, vk::DebugReportObjectTypeEXT::eFence, name);
		}

		void setEventName(vk::Device device, vk::Event _event, const char * name)
		{
			setObjectName(device, (uint64_t)(VkEvent)_event, vk::DebugReportObjectTypeEXT::eEvent, name);
		}
	};
}

