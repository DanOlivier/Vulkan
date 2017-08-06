/*
* Vulkan device class
*
* Encapsulates a physical Vulkan device and it's logical representation
*
* Copyright (C) 2016-2017 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#pragma once

#include <exception>
#include <assert.h>
#include <algorithm>
#include "vulkan/vulkan.hpp"
#include "VulkanTools.h"
#include "VulkanBuffer.hpp"

namespace vks
{	
	struct VulkanDevice
	{
		/** @brief Physical device representation */
		vk::PhysicalDevice physicalDevice;
		/** @brief Logical device representation (application's view of the device) */
		vk::Device logicalDevice;
		/** @brief Properties of the physical device including limits that the application can check against */
		vk::PhysicalDeviceProperties properties;
		/** @brief Features of the physical device that an application can use to check if a feature is supported */
		vk::PhysicalDeviceFeatures features;
		/** @brief Features that have been enabled for use on the physical device */
		vk::PhysicalDeviceFeatures enabledFeatures;
		/** @brief Memory types and heaps of the physical device */
		vk::PhysicalDeviceMemoryProperties memoryProperties;
		/** @brief Queue family properties of the physical device */
		std::vector<vk::QueueFamilyProperties> queueFamilyProperties;
		/** @brief List of extensions supported by the device */
		std::vector<std::string> supportedExtensions;

		/** @brief Default command pool for the graphics queue family index */
		vk::CommandPool commandPool;

		/** @brief Set to true when the debug marker extension is detected */
		bool enableDebugMarkers = false;

		/** @brief Contains queue family indices */
		struct
		{
			uint32_t graphics;
			uint32_t compute;
			uint32_t transfer;
		} queueFamilyIndices;

		/**  @brief Typecast to vk::Device */
		operator vk::Device() { return logicalDevice; };

		/**
		* Default constructor
		*
		* @param physicalDevice Physical device that is to be used
		*/
		VulkanDevice(vk::PhysicalDevice physicalDevice)
		{
			assert(physicalDevice);
			this->physicalDevice = physicalDevice;

			// Store Properties features, limits and properties of the physical device for later use
			// Device properties also contain limits and sparse properties
			properties = physicalDevice.getProperties();
			// Features should be checked by the examples before using them
			features = physicalDevice.getFeatures();
			// Memory properties are used regularly for creating all kinds of buffers
			memoryProperties = physicalDevice.getMemoryProperties();
			// Queue family properties, used for setting up requested queues upon device creation
			queueFamilyProperties = physicalDevice.getQueueFamilyProperties();

			// Get list of supported extensions
			std::vector<vk::ExtensionProperties> extensions = physicalDevice.enumerateDeviceExtensionProperties();
			for (auto ext : extensions)
			{
				supportedExtensions.push_back(ext.extensionName);
			}
		}

		/** 
		* Default destructor
		*
		* @note Frees the logical device
		*/
		~VulkanDevice()
		{
			if (commandPool)
			{
				logicalDevice.destroyCommandPool(commandPool);
			}
			if (logicalDevice)
			{
				logicalDevice.destroy();
			}
		}

		/**
		* Get the index of a memory type that has all the requested property bits set
		*
		* @param typeBits Bitmask with bits set for each memory type supported by the resource to request for (from vk::MemoryRequirements)
		* @param properties Bitmask of properties for the memory type to request
		* @param (Optional) memTypeFound Pointer to a bool that is set to true if a matching memory type has been found
		* 
		* @return Index of the requested memory type
		*
		* @throw Throws an exception if memTypeFound is null and no memory type could be found that supports the requested properties
		*/
		uint32_t getMemoryType(uint32_t typeBits, vk::MemoryPropertyFlags properties, vk::Bool32 *memTypeFound = nullptr)
		{
			for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; i++)
			{
				if ((typeBits & 1) == 1)
				{
					if ((memoryProperties.memoryTypes[i].propertyFlags & properties) == properties)
					{
						if (memTypeFound)
						{
							*memTypeFound = true;
						}
						return i;
					}
				}
				typeBits >>= 1;
			}

#if defined(__ANDROID__)
			//todo : Exceptions are disabled by default on Android (need to add LOCAL_CPP_FEATURES += exceptions to Android.mk), so for now just return zero
			if (memTypeFound)
			{
				*memTypeFound = false;
			}
			return 0;
#else
			if (memTypeFound)
			{
				*memTypeFound = false;
				return 0;
			}
			else
			{
				throw std::runtime_error("Could not find a matching memory type");
			}
#endif
		}

		/**
		* Get the index of a queue family that supports the requested queue flags
		*
		* @param queueFlags Queue flags to find a queue family index for
		*
		* @return Index of the queue family index that matches the flags
		*
		* @throw Throws an exception if no queue family index could be found that supports the requested flags
		*/
		uint32_t getQueueFamilyIndex(vk::QueueFlagBits queueFlags)
		{
			// Dedicated queue for compute
			// Try to find a queue family index that supports compute but not graphics
			if (vk::QueueFlags(queueFlags) & vk::QueueFlagBits::eCompute)
			{
				for (uint32_t i = 0; i < static_cast<uint32_t>(queueFamilyProperties.size()); i++)
				{
					if ((queueFamilyProperties[i].queueFlags & queueFlags) && 
						~(queueFamilyProperties[i].queueFlags & vk::QueueFlagBits::eGraphics))
					{
						return i;
						break;
					}
				}
			}

			// Dedicated queue for transfer
			// Try to find a queue family index that supports transfer but not graphics and compute
			if (vk::QueueFlags(queueFlags) & vk::QueueFlagBits::eTransfer)
			{
				for (uint32_t i = 0; i < static_cast<uint32_t>(queueFamilyProperties.size()); i++)
				{
					if ((queueFamilyProperties[i].queueFlags & queueFlags) && 
						~(queueFamilyProperties[i].queueFlags & vk::QueueFlagBits::eGraphics) && 
						~(queueFamilyProperties[i].queueFlags & vk::QueueFlagBits::eCompute))
					{
						return i;
						break;
					}
				}
			}

			// For other queue types or if no separate compute queue is present, return the first one to support the requested flags
			for (uint32_t i = 0; i < static_cast<uint32_t>(queueFamilyProperties.size()); i++)
			{
				if (queueFamilyProperties[i].queueFlags & queueFlags)
				{
					return i;
					break;
				}
			}

#if defined(__ANDROID__)
			//todo : Exceptions are disabled by default on Android (need to add LOCAL_CPP_FEATURES += exceptions to Android.mk), so for now just return zero
			return 0;
#else
			throw std::runtime_error("Could not find a matching queue family index");
#endif
		}

		/**
		* Create the logical device based on the assigned physical device, also gets default queue family indices
		*
		* @param enabledFeatures Can be used to enable certain features upon device creation
		* @param useSwapChain Set to false for headless rendering to omit the swapchain device extensions
		* @param requestedQueueTypes Bit flags specifying the queue types to be requested from the device  
		*
		* @return vk::Result of the device creation call
		*/
		vk::Result createLogicalDevice(vk::PhysicalDeviceFeatures enabledFeatures, 
			std::vector<const char*> enabledExtensions, bool useSwapChain = true, 
			vk::QueueFlags requestedQueueTypes = vk::QueueFlagBits::eGraphics | vk::QueueFlagBits::eCompute)
		{			
			// Desired queues need to be requested upon logical device creation
			// Due to differing queue family configurations of Vulkan implementations this can be a bit tricky, especially if the application
			// requests different queue types

			std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos{};

			// Get queue family indices for the requested queue family types
			// Note that the indices may overlap depending on the implementation

			const float defaultQueuePriority(0.0f);

			// Graphics queue
			if (requestedQueueTypes & vk::QueueFlagBits::eGraphics)
			{
				queueFamilyIndices.graphics = getQueueFamilyIndex(vk::QueueFlagBits::eGraphics);
				vk::DeviceQueueCreateInfo queueInfo{};

				queueInfo.queueFamilyIndex = queueFamilyIndices.graphics;
				queueInfo.queueCount = 1;
				queueInfo.pQueuePriorities = &defaultQueuePriority;
				queueCreateInfos.push_back(queueInfo);
			}
			else
			{
				queueFamilyIndices.graphics = 0;
			}

			// Dedicated compute queue
			if (requestedQueueTypes & vk::QueueFlagBits::eCompute)
			{
				queueFamilyIndices.compute = getQueueFamilyIndex(vk::QueueFlagBits::eCompute);
				if (queueFamilyIndices.compute != queueFamilyIndices.graphics)
				{
					// If compute family index differs, we need an additional queue create info for the compute queue
					vk::DeviceQueueCreateInfo queueInfo{};

					queueInfo.queueFamilyIndex = queueFamilyIndices.compute;
					queueInfo.queueCount = 1;
					queueInfo.pQueuePriorities = &defaultQueuePriority;
					queueCreateInfos.push_back(queueInfo);
				}
			}
			else
			{
				// Else we use the same queue
				queueFamilyIndices.compute = queueFamilyIndices.graphics;
			}

			// Dedicated transfer queue
			if (requestedQueueTypes & vk::QueueFlagBits::eTransfer)
			{
				queueFamilyIndices.transfer = getQueueFamilyIndex(vk::QueueFlagBits::eTransfer);
				if ((queueFamilyIndices.transfer != queueFamilyIndices.graphics) && 
					(queueFamilyIndices.transfer != queueFamilyIndices.compute))
				{
					// If compute family index differs, we need an additional queue create info for the compute queue
					vk::DeviceQueueCreateInfo queueInfo{};

					queueInfo.queueFamilyIndex = queueFamilyIndices.transfer;
					queueInfo.queueCount = 1;
					queueInfo.pQueuePriorities = &defaultQueuePriority;
					queueCreateInfos.push_back(queueInfo);
				}
			}
			else
			{
				// Else we use the same queue
				queueFamilyIndices.transfer = queueFamilyIndices.graphics;
			}

			// Create the logical device representation
			std::vector<const char*> deviceExtensions(enabledExtensions);
			if (useSwapChain)
			{
				// If the device will be used for presenting to a display via a swapchain we need to request the swapchain extension
				deviceExtensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
			}

			vk::DeviceCreateInfo deviceCreateInfo = {};

			deviceCreateInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());;
			deviceCreateInfo.pQueueCreateInfos = queueCreateInfos.data();
			deviceCreateInfo.pEnabledFeatures = &enabledFeatures;

			// Enable the debug marker extension if it is present (likely meaning a debugging tool is present)
			if (extensionSupported(VK_EXT_DEBUG_MARKER_EXTENSION_NAME))
			{
				deviceExtensions.push_back(VK_EXT_DEBUG_MARKER_EXTENSION_NAME);
				enableDebugMarkers = true;
			}

			if (deviceExtensions.size() > 0)
			{
				deviceCreateInfo.enabledExtensionCount = (uint32_t)deviceExtensions.size();
				deviceCreateInfo.ppEnabledExtensionNames = deviceExtensions.data();
			}

			logicalDevice = physicalDevice.createDevice(deviceCreateInfo);

			// Create a default command pool for graphics command buffers
			commandPool = createCommandPool(queueFamilyIndices.graphics);

			this->enabledFeatures = enabledFeatures;

			return vk::Result::eSuccess;
		}

		/**
		* Create a buffer on the device
		*
		* @param usageFlags Usage flag bitmask for the buffer (i.e. index, vertex, uniform buffer)
		* @param memoryPropertyFlags Memory properties for this buffer (i.e. device local, host visible, coherent)
		* @param size Size of the buffer in byes
		* @param buffer Pointer to the buffer handle acquired by the function
		* @param memory Pointer to the memory handle acquired by the function
		* @param data Pointer to the data that should be copied to the buffer after creation (optional, if not set, no data is copied over)
		*
		* @return vk::Result::eSuccess if buffer handle and memory have been created and (optionally passed) data has been copied
		*/
		vk::Result createBuffer(vk::BufferUsageFlags usageFlags, vk::MemoryPropertyFlags memoryPropertyFlags, vk::DeviceSize size, vk::Buffer *buffer, vk::DeviceMemory *memory, void *data = nullptr)
		{
			// Create the buffer handle
			vk::BufferCreateInfo bufferCreateInfo = vks::initializers::bufferCreateInfo(usageFlags, size);
			bufferCreateInfo.sharingMode = vk::SharingMode::eExclusive;
			*buffer = logicalDevice.createBuffer(bufferCreateInfo);

			// Create the memory backing up the buffer handle
			vk::MemoryRequirements memReqs;
			vk::MemoryAllocateInfo memAlloc = vks::initializers::memoryAllocateInfo();
			memReqs = logicalDevice.getBufferMemoryRequirements(*buffer);
			memAlloc.allocationSize = memReqs.size;
			// Find a memory type index that fits the properties of the buffer
			memAlloc.memoryTypeIndex = getMemoryType(memReqs.memoryTypeBits, memoryPropertyFlags);
			*memory = logicalDevice.allocateMemory(memAlloc);
			
			// If a pointer to the buffer data has been passed, map the buffer and copy over the data
			if (data != nullptr)
			{
				void *mapped;
				mapped = logicalDevice.mapMemory(*memory, 0, size, vk::MemoryMapFlags());
				memcpy(mapped, data, size);
				// If host coherency hasn't been requested, do a manual flush to make writes visible
				if (~(memoryPropertyFlags & vk::MemoryPropertyFlagBits::eHostCoherent))
				{
					vk::MappedMemoryRange mappedRange = vks::initializers::mappedMemoryRange();
					mappedRange.memory = *memory;
					mappedRange.offset = 0;
					mappedRange.size = size;
					logicalDevice.flushMappedMemoryRanges(mappedRange);
				}
				logicalDevice.unmapMemory(*memory);
			}

			// Attach the memory to the buffer object
			logicalDevice.bindBufferMemory(*buffer, *memory, 0);

			return vk::Result::eSuccess;
		}

		/**
		* Create a buffer on the device
		*
		* @param usageFlags Usage flag bitmask for the buffer (i.e. index, vertex, uniform buffer)
		* @param memoryPropertyFlags Memory properties for this buffer (i.e. device local, host visible, coherent)
		* @param buffer Pointer to a vk::Vulkan buffer object
		* @param size Size of the buffer in byes
		* @param data Pointer to the data that should be copied to the buffer after creation (optional, if not set, no data is copied over)
		*
		* @return vk::Result::eSuccess if buffer handle and memory have been created and (optionally passed) data has been copied
		*/
		vk::Result createBuffer(vk::BufferUsageFlags usageFlags, vk::MemoryPropertyFlags memoryPropertyFlags, vks::Buffer *buffer, vk::DeviceSize size, void *data = nullptr)
		{
			buffer->device = logicalDevice;

			// Create the buffer handle
			vk::BufferCreateInfo bufferCreateInfo = vks::initializers::bufferCreateInfo(usageFlags, size);
			buffer->buffer = logicalDevice.createBuffer(bufferCreateInfo);

			// Create the memory backing up the buffer handle
			vk::MemoryRequirements memReqs;
			vk::MemoryAllocateInfo memAlloc = vks::initializers::memoryAllocateInfo();
			memReqs = logicalDevice.getBufferMemoryRequirements(buffer->buffer);
			memAlloc.allocationSize = memReqs.size;
			// Find a memory type index that fits the properties of the buffer
			memAlloc.memoryTypeIndex = getMemoryType(memReqs.memoryTypeBits, memoryPropertyFlags);
			buffer->memory = logicalDevice.allocateMemory(memAlloc);

			buffer->alignment = memReqs.alignment;
			buffer->size = memAlloc.allocationSize;
			buffer->usageFlags = usageFlags;
			buffer->memoryPropertyFlags = memoryPropertyFlags;

			// If a pointer to the buffer data has been passed, map the buffer and copy over the data
			if (data != nullptr)
			{
				buffer->map();
				memcpy(buffer->mapped, data, size);
				buffer->unmap();
			}

			// Initialize a default descriptor that covers the whole buffer size
			buffer->setupDescriptor();

			// Attach the memory to the buffer object
			buffer->bind();
		}

		/**
		* Copy buffer data from src to dst using vk::CmdCopyBuffer
		* 
		* @param src Pointer to the source buffer to copy from
		* @param dst Pointer to the destination buffer to copy tp
		* @param queue Pointer
		* @param copyRegion (Optional) Pointer to a copy region, if NULL, the whole buffer is copied
		*
		* @note Source and destionation pointers must have the approriate transfer usage flags set (TRANSFER_SRC / TRANSFER_DST)
		*/
		void copyBuffer(vks::Buffer *src, vks::Buffer *dst, vk::Queue queue, vk::BufferCopy *copyRegion = nullptr)
		{
			assert(dst->size <= src->size);
			assert(src->buffer && src->buffer);
			vk::CommandBuffer copyCmd = createCommandBuffer(vk::CommandBufferLevel::ePrimary, true);
			vk::BufferCopy bufferCopy{};
			if (copyRegion == nullptr)
			{
				bufferCopy.size = src->size;
			}
			else
			{
				bufferCopy = *copyRegion;
			}

			copyCmd.copyBuffer(src->buffer, dst->buffer, bufferCopy);

			flushCommandBuffer(copyCmd, queue);
		}

		/** 
		* Create a command pool for allocation command buffers from
		* 
		* @param queueFamilyIndex Family index of the queue to create the command pool for
		* @param createFlags (Optional) Command pool creation flags (Defaults to vk::CommandPoolCreateFlagBits::eResetCommandBuffer)
		*
		* @note Command buffers allocated from the created pool can only be submitted to a queue with the same family index
		*
		* @return A handle to the created command buffer
		*/
		vk::CommandPool createCommandPool(uint32_t queueFamilyIndex, vk::CommandPoolCreateFlags createFlags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer)
		{
			vk::CommandPoolCreateInfo cmdPoolInfo = {};

			cmdPoolInfo.queueFamilyIndex = queueFamilyIndex;
			cmdPoolInfo.flags = createFlags;
			vk::CommandPool cmdPool;
			cmdPool = logicalDevice.createCommandPool(cmdPoolInfo);
			return cmdPool;
		}

		/**
		* Allocate a command buffer from the command pool
		*
		* @param level Level of the new command buffer (primary or secondary)
		* @param (Optional) begin If true, recording on the new command buffer will be started (vkBeginCommandBuffer) (Defaults to false)
		*
		* @return A handle to the allocated command buffer
		*/
		vk::CommandBuffer createCommandBuffer(vk::CommandBufferLevel level, bool begin = false)
		{
			vk::CommandBufferAllocateInfo cmdBufAllocateInfo = vks::initializers::commandBufferAllocateInfo(commandPool, level, 1);

			vk::CommandBuffer cmdBuffer = logicalDevice.allocateCommandBuffers(cmdBufAllocateInfo)[0];

			// If requested, also start recording for the new command buffer
			if (begin)
			{
				vk::CommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();
				cmdBuffer.begin(cmdBufInfo);
			}

			return cmdBuffer;
		}

		/**
		* Finish command buffer recording and submit it to a queue
		*
		* @param commandBuffer Command buffer to flush
		* @param queue Queue to submit the command buffer to 
		* @param free (Optional) Free the command buffer once it has been submitted (Defaults to true)
		*
		* @note The queue that the command buffer is submitted to must be from the same family index as the pool it was allocated from
		* @note Uses a fence to ensure command buffer has finished executing
		*/
		void flushCommandBuffer(vk::CommandBuffer commandBuffer, vk::Queue queue, bool free = true)
		{
			if (!commandBuffer)
			{
				return;
			}

			commandBuffer.end();

			vk::SubmitInfo submitInfo = vks::initializers::submitInfo();
			submitInfo.commandBufferCount = 1;
			submitInfo.pCommandBuffers = &commandBuffer;

			// Create fence to ensure that the command buffer has finished executing
			vk::FenceCreateInfo fenceInfo = vks::initializers::fenceCreateInfo();
			vk::Fence fence;
			fence = logicalDevice.createFence(fenceInfo);
			
			// Submit to the queue
			queue.submit(submitInfo, fence);
			// Wait for the fence to signal that command buffer has finished executing
			logicalDevice.waitForFences(fence, VK_TRUE, DEFAULT_FENCE_TIMEOUT);

			logicalDevice.destroyFence(fence);

			if (free)
			{
				logicalDevice.freeCommandBuffers(commandPool, commandBuffer);
			}
		}

		/**
		* Check if an extension is supported by the (physical device)
		*
		* @param extension Name of the extension to check
		*
		* @return True if the extension is supported (present in the list read at device creation time)
		*/
		bool extensionSupported(std::string extension)
		{
			return (std::find(supportedExtensions.begin(), supportedExtensions.end(), extension) != supportedExtensions.end());
		}

	};
}
