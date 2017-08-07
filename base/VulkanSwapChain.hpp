/*
* Class wrapping access to the swap chain
* 
* A swap chain is a collection of framebuffers used for rendering and presentation to the windowing system
*
* Copyright (C) 2016-2017 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#pragma once

#include <stdlib.h>
#include <string>
#include <assert.h>
#include <stdio.h>
#include <vector>

#include <vulkan/vulkan.hpp>
#include "VulkanTools.h"

#ifdef __ANDROID__
#include "VulkanAndroid.h"
#endif

// Macro to get a procedure address based on a vulkan instance
#define GET_INSTANCE_PROC_ADDR(inst, entrypoint)                        \
{                                                                       \
	fp##entrypoint = reinterpret_cast<PFN_vk##entrypoint>(vkGetInstanceProcAddr(inst, "vk"#entrypoint)); \
	if (fp##entrypoint == NULL)                                         \
	{																    \
		exit(1);                                                        \
	}                                                                   \
}

// Macro to get a procedure address based on a vulkan device
#define GET_DEVICE_PROC_ADDR(dev, entrypoint)                           \
{                                                                       \
	fp##entrypoint = reinterpret_cast<PFN_vk##entrypoint>(vkGetDeviceProcAddr(dev, "vk"#entrypoint));   \
	if (fp##entrypoint == NULL)                                         \
	{																    \
		exit(1);                                                        \
	}                                                                   \
}

typedef struct _SwapChainBuffers {
	vk::Image image;
	vk::ImageView view;
} SwapChainBuffer;

class VulkanSwapChain
{
private: 
	vk::Instance instance;
	vk::Device device;
	vk::PhysicalDevice physicalDevice;
	vk::SurfaceKHR surface;
	// Function pointers
	PFN_vkGetPhysicalDeviceSurfaceSupportKHR fpGetPhysicalDeviceSurfaceSupportKHR;
	PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR fpGetPhysicalDeviceSurfaceCapabilitiesKHR; 
	PFN_vkGetPhysicalDeviceSurfaceFormatsKHR fpGetPhysicalDeviceSurfaceFormatsKHR;
	PFN_vkGetPhysicalDeviceSurfacePresentModesKHR fpGetPhysicalDeviceSurfacePresentModesKHR;
	PFN_vkCreateSwapchainKHR fpCreateSwapchainKHR;
	PFN_vkDestroySwapchainKHR fpDestroySwapchainKHR;
	PFN_vkGetSwapchainImagesKHR fpGetSwapchainImagesKHR;
	PFN_vkAcquireNextImageKHR fpAcquireNextImageKHR;
	PFN_vkQueuePresentKHR fpQueuePresentKHR;
public:
	vk::Format colorFormat;
	vk::ColorSpaceKHR colorSpace;
	/** @brief Handle to the current swap chain, required for recreation */
	vk::SwapchainKHR swapChain;	
	uint32_t imageCount;
	std::vector<vk::Image> images;
	std::vector<SwapChainBuffer> buffers;
	/** @brief Queue family index of the detected graphics and presenting device queue */
	uint32_t queueNodeIndex = UINT32_MAX;

	/** @brief Creates the platform specific surface abstraction of the native platform window used for presentation */	
#if defined(VK_USE_PLATFORM_WIN32_KHR)
	void initSurface(void* platformHandle, void* platformWindow)
#elif defined(VK_USE_PLATFORM_ANDROID_KHR)
	void initSurface(ANativeWindow* window)
#elif defined(VK_USE_PLATFORM_WAYLAND_KHR)
	void initSurface(wl_display *display, wl_surface *window)
#elif defined(VK_USE_PLATFORM_XCB_KHR)
	void initSurface(xcb_connection_t* connection, xcb_window_t window)
#elif (defined(VK_USE_PLATFORM_IOS_MVK) || defined(VK_USE_PLATFORM_MACOS_MVK))
	void initSurface(void* view)
#elif defined(_DIRECT2DISPLAY)
	void initSurface(uint32_t width, uint32_t height)
#endif
	{
		vk::Result err = vk::Result::eSuccess;

		// Create the os-specific surface
#if defined(VK_USE_PLATFORM_WIN32_KHR)
		vk::Win32SurfaceCreateInfoKHR surfaceCreateInfo = {};

		surfaceCreateInfo.hinstance = (HINSTANCE)platformHandle;
		surfaceCreateInfo.hwnd = (HWND)platformWindow;
		surface = instance.createWin32SurfaceKHR(surfaceCreateInfo);
#elif defined(VK_USE_PLATFORM_ANDROID_KHR)
		vk::AndroidSurfaceCreateInfoKHR surfaceCreateInfo = {};

		surfaceCreateInfo.window = window;
		surface = instancecreateAndroidSurfaceKHR(surfaceCreateInfo);
#elif defined(VK_USE_PLATFORM_IOS_MVK)
		vk::IOSSurfaceCreateInfoMVK surfaceCreateInfo = {};
		//surfaceCreateInfo.flags = 0;
		surfaceCreateInfo.pView = view;
		surface = instance.createIOSSurfaceMVK(surfaceCreateInfo);
#elif defined(VK_USE_PLATFORM_MACOS_MVK)
		vk::MacOSSurfaceCreateInfoMVK surfaceCreateInfo = {};
		//surfaceCreateInfo.flags = 0;
		surfaceCreateInfo.pView = view;
		surface = instance.createMacOSSurfaceMVK(surfaceCreateInfo);
#elif defined(_DIRECT2DISPLAY)
		createDirect2DisplaySurface(width, height);
#elif defined(VK_USE_PLATFORM_WAYLAND_KHR)
		vk::WaylandSurfaceCreateInfoKHR surfaceCreateInfo = {};

		surfaceCreateInfo.display = display;
		surfaceCreateInfo.surface = window;
		surface = instance.ceateWaylandSurfaceKHR(surfaceCreateInfo);
#elif defined(VK_USE_PLATFORM_XCB_KHR)
		vk::XcbSurfaceCreateInfoKHR surfaceCreateInfo = {};

		surfaceCreateInfo.connection = connection;
		surfaceCreateInfo.window = window;
		surface = instance.createXcbSurfaceKHR(surfaceCreateInfo);
#endif

		if (err != vk::Result::eSuccess) {
			vks::tools::exitFatal("Could not create surface!", "Fatal error");
		}

		// Get available queue family properties
		std::vector<vk::QueueFamilyProperties> queueProps = physicalDevice.getQueueFamilyProperties();
		uint32_t queueCount = queueProps.size();

		// Iterate over each queue to learn whether it supports presenting:
		// Find a queue with present support
		// Will be used to present the swap chain images to the windowing system
		std::vector<vk::Bool32> supportsPresent(queueCount);
		for (uint32_t i = 0; i < queueCount; i++) 
		{
			supportsPresent[i] = physicalDevice.getSurfaceSupportKHR(i, surface);
		}

		// Search for a graphics and a present queue in the array of queue
		// families, try to find one that supports both
		uint32_t graphicsQueueNodeIndex = UINT32_MAX;
		uint32_t presentQueueNodeIndex = UINT32_MAX;
		for (uint32_t i = 0; i < queueCount; i++) 
		{
			if (queueProps[i].queueFlags & vk::QueueFlagBits::eGraphics) 
			{
				if (graphicsQueueNodeIndex == UINT32_MAX) 
				{
					graphicsQueueNodeIndex = i;
				}

				if (supportsPresent[i] == VK_TRUE) 
				{
					graphicsQueueNodeIndex = i;
					presentQueueNodeIndex = i;
					break;
				}
			}
		}
		if (presentQueueNodeIndex == UINT32_MAX) 
		{	
			// If there's no queue that supports both present and graphics
			// try to find a separate present queue
			for (uint32_t i = 0; i < queueCount; ++i) 
			{
				if (supportsPresent[i] == VK_TRUE) 
				{
					presentQueueNodeIndex = i;
					break;
				}
			}
		}

		// Exit if either a graphics or a presenting queue hasn't been found
		if (graphicsQueueNodeIndex == UINT32_MAX || presentQueueNodeIndex == UINT32_MAX) 
		{
			vks::tools::exitFatal("Could not find a graphics and/or presenting queue!", "Fatal error");
		}

		// todo : Add support for separate graphics and presenting queue
		if (graphicsQueueNodeIndex != presentQueueNodeIndex) 
		{
			vks::tools::exitFatal("Separate graphics and presenting queues are not supported yet!", "Fatal error");
		}

		queueNodeIndex = graphicsQueueNodeIndex;

		// Get list of supported surface formats
		std::vector<vk::SurfaceFormatKHR> surfaceFormats = physicalDevice.getSurfaceFormatsKHR(surface);
		uint32_t formatCount = surfaceFormats.size();
		assert(formatCount > 0);

		// If the surface format list only includes one entry with vk::Format::eUndefined,
		// there is no preferered format, so we assume vk::Format::eB8G8R8A8Unorm
		if ((formatCount == 1) && (surfaceFormats[0].format == vk::Format::eUndefined))
		{
			colorFormat = vk::Format::eB8G8R8A8Unorm;
			colorSpace = surfaceFormats[0].colorSpace;
		}
		else
		{
			// iterate over the list of available surface format and
			// check for the presence of vk::Format::eB8G8R8A8Unorm
			bool found_B8G8R8A8_UNORM = false;
			for (auto&& surfaceFormat : surfaceFormats)
			{
				if (surfaceFormat.format == vk::Format::eB8G8R8A8Unorm)
				{
					colorFormat = surfaceFormat.format;
					colorSpace = surfaceFormat.colorSpace;
					found_B8G8R8A8_UNORM = true;
					break;
				}
			}

			// in case vk::Format::eB8G8R8A8Unorm is not available
			// select the first available color format
			if (!found_B8G8R8A8_UNORM)
			{
				colorFormat = surfaceFormats[0].format;
				colorSpace = surfaceFormats[0].colorSpace;
			}
		}

	}

	/**
	* Set instance, physical and logical device to use for the swapchain and get all required function pointers
	* 
	* @param instance Vulkan instance to use
	* @param physicalDevice Physical device used to query properties and formats relevant to the swapchain
	* @param device Logical representation of the device to create the swapchain for
	*
	*/
	void connect(vk::Instance instance, vk::PhysicalDevice physicalDevice, vk::Device device)
	{
		this->instance = instance;
		this->physicalDevice = physicalDevice;
		this->device = device;
		GET_INSTANCE_PROC_ADDR(instance, GetPhysicalDeviceSurfaceSupportKHR);
		GET_INSTANCE_PROC_ADDR(instance, GetPhysicalDeviceSurfaceCapabilitiesKHR);
		GET_INSTANCE_PROC_ADDR(instance, GetPhysicalDeviceSurfaceFormatsKHR);
		GET_INSTANCE_PROC_ADDR(instance, GetPhysicalDeviceSurfacePresentModesKHR);
		GET_DEVICE_PROC_ADDR(device, CreateSwapchainKHR);
		GET_DEVICE_PROC_ADDR(device, DestroySwapchainKHR);
		GET_DEVICE_PROC_ADDR(device, GetSwapchainImagesKHR);
		GET_DEVICE_PROC_ADDR(device, AcquireNextImageKHR);
		GET_DEVICE_PROC_ADDR(device, QueuePresentKHR);
	}

	/** 
	* Create the swapchain and get it's images with given width and height
	* 
	* @param width Pointer to the width of the swapchain (may be adjusted to fit the requirements of the swapchain)
	* @param height Pointer to the height of the swapchain (may be adjusted to fit the requirements of the swapchain)
	* @param vsync (Optional) Can be used to force vsync'd rendering (by using vk::PresentModeKHR::eFifo as presentation mode)
	*/
	void create(uint32_t *width, uint32_t *height, bool vsync = false)
	{
		vk::SwapchainKHR oldSwapchain = swapChain;

		// Get physical device surface properties and formats
		vk::SurfaceCapabilitiesKHR surfCaps = physicalDevice.getSurfaceCapabilitiesKHR(surface);

		// Get available present modes
		std::vector<vk::PresentModeKHR> presentModes = physicalDevice.getSurfacePresentModesKHR(surface);
		uint32_t presentModeCount = presentModes.size();
		assert(presentModeCount > 0);

		vk::Extent2D swapchainExtent = {};
		// If width (and height) equals the special value 0xFFFFFFFF, the size of the surface will be set by the swapchain
		if (surfCaps.currentExtent.width == (uint32_t)-1)
		{
			// If the surface size is undefined, the size is set to
			// the size of the images requested.
			swapchainExtent.width = *width;
			swapchainExtent.height = *height;
		}
		else
		{
			// If the surface size is defined, the swap chain size must match
			swapchainExtent = surfCaps.currentExtent;
			*width = surfCaps.currentExtent.width;
			*height = surfCaps.currentExtent.height;
		}


		// Select a present mode for the swapchain

		// The vk::PresentModeKHR::eFifo mode must always be present as per spec
		// This mode waits for the vertical blank ("v-sync")
		vk::PresentModeKHR swapchainPresentMode = vk::PresentModeKHR::eFifo;

		// If v-sync is not requested, try to find a mailbox mode
		// It's the lowest latency non-tearing present mode available
		if (!vsync)
		{
			for (size_t i = 0; i < presentModeCount; i++)
			{
				if (presentModes[i] == vk::PresentModeKHR::eMailbox)
				{
					swapchainPresentMode = vk::PresentModeKHR::eMailbox;
					break;
				}
				if ((swapchainPresentMode != vk::PresentModeKHR::eMailbox) && (presentModes[i] == vk::PresentModeKHR::eImmediate))
				{
					swapchainPresentMode = vk::PresentModeKHR::eImmediate;
				}
			}
		}

		// Determine the number of images
		uint32_t desiredNumberOfSwapchainImages = surfCaps.minImageCount + 1;
		if ((surfCaps.maxImageCount > 0) && (desiredNumberOfSwapchainImages > surfCaps.maxImageCount))
		{
			desiredNumberOfSwapchainImages = surfCaps.maxImageCount;
		}

		// Find the transformation of the surface
		vk::SurfaceTransformFlagBitsKHR preTransform;
		if (surfCaps.supportedTransforms & vk::SurfaceTransformFlagBitsKHR::eIdentity)
		{
			// We prefer a non-rotated transform
			preTransform = vk::SurfaceTransformFlagBitsKHR::eIdentity;
		}
		else
		{
			preTransform = surfCaps.currentTransform;
		}

		// Find a supported composite alpha format (not all devices support alpha opaque)
		vk::CompositeAlphaFlagBitsKHR compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;
		// Simply select the first composite alpha format available
		std::vector<vk::CompositeAlphaFlagBitsKHR> compositeAlphaFlags = {
			vk::CompositeAlphaFlagBitsKHR::eOpaque,
			vk::CompositeAlphaFlagBitsKHR::ePreMultiplied,
			vk::CompositeAlphaFlagBitsKHR::ePostMultiplied,
			vk::CompositeAlphaFlagBitsKHR::eInherit,
		};
		for (auto& compositeAlphaFlag : compositeAlphaFlags) {
			if (surfCaps.supportedCompositeAlpha & compositeAlphaFlag) {
				compositeAlpha = compositeAlphaFlag;
				break;
			};
		}

		vk::SwapchainCreateInfoKHR swapchainCI = {};
		swapchainCI.surface = surface;
		swapchainCI.minImageCount = desiredNumberOfSwapchainImages;
		swapchainCI.imageFormat = colorFormat;
		swapchainCI.imageColorSpace = colorSpace;
		swapchainCI.imageExtent = vk::Extent2D{ swapchainExtent.width, swapchainExtent.height };
		swapchainCI.imageUsage = vk::ImageUsageFlagBits::eColorAttachment;
		swapchainCI.preTransform = preTransform;
		swapchainCI.imageArrayLayers = 1;
		swapchainCI.imageSharingMode = vk::SharingMode::eExclusive;
		swapchainCI.queueFamilyIndexCount = 0;
		swapchainCI.pQueueFamilyIndices = NULL;
		swapchainCI.presentMode = swapchainPresentMode;
		swapchainCI.oldSwapchain = oldSwapchain;
		// Setting clipped to VK_TRUE allows the implementation to discard rendering outside of the surface area
		swapchainCI.clipped = VK_TRUE;
		swapchainCI.compositeAlpha = compositeAlpha;

		// Set additional usage flag for blitting from the swapchain images if supported
		vk::FormatProperties formatProps = physicalDevice.getFormatProperties(colorFormat);
		if (formatProps.optimalTilingFeatures & vk::FormatFeatureFlagBits::eBlitDst) {
			swapchainCI.imageUsage |= vk::ImageUsageFlagBits::eTransferSrc;
		}

		swapChain = device.createSwapchainKHR(swapchainCI);

		// If an existing swap chain is re-created, destroy the old swap chain
		// This also cleans up all the presentable images
		if (oldSwapchain) 
		{ 
			for (uint32_t i = 0; i < imageCount; i++)
			{
				device.destroyImageView(buffers[i].view);
			}
			device.destroySwapchainKHR(oldSwapchain);
		}

		// Get the swap chain images
		images = device.getSwapchainImagesKHR(swapChain);
		imageCount = images.size();

		// Get the swap chain buffers containing the image and imageview
		buffers.resize(imageCount);
		for (uint32_t i = 0; i < imageCount; i++)
		{
			vk::ImageViewCreateInfo colorAttachmentView = {};
			colorAttachmentView.format = colorFormat;
			colorAttachmentView.components = {
				vk::ComponentSwizzle::eR,
				vk::ComponentSwizzle::eG,
				vk::ComponentSwizzle::eB,
				vk::ComponentSwizzle::eA
			};
			colorAttachmentView.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
			colorAttachmentView.subresourceRange.baseMipLevel = 0;
			colorAttachmentView.subresourceRange.levelCount = 1;
			colorAttachmentView.subresourceRange.baseArrayLayer = 0;
			colorAttachmentView.subresourceRange.layerCount = 1;
			colorAttachmentView.viewType = vk::ImageViewType::e2D;
			colorAttachmentView.flags = vk::ImageViewCreateFlags();

			buffers[i].image = images[i];

			colorAttachmentView.image = buffers[i].image;

			buffers[i].view = device.createImageView(colorAttachmentView);
		}
	}

	/** 
	* Acquires the next image in the swap chain
	*
	* @param presentCompleteSemaphore (Optional) Semaphore that is signaled when the image is ready for use
	* @param imageIndex Pointer to the image index that will be increased if the next image could be acquired
	*
	* @note The function will always wait until the next image has been acquired by setting timeout to UINT64_MAX
	*
	* @return vk::Result of the image acquisition
	*/
	uint32_t acquireNextImage(vk::Semaphore presentCompleteSemaphore)
	{
		// By setting timeout to UINT64_MAX we will always wait until the next image has been acquired or an actual error is thrown
		// With that we don't have to handle vk::Result::eNotReady
		return device.acquireNextImageKHR(swapChain, UINT64_MAX, presentCompleteSemaphore, (vk::Fence)nullptr).value;
	}

	/**
	* Queue an image for presentation
	*
	* @param queue Presentation queue for presenting the image
	* @param imageIndex Index of the swapchain image to queue for presentation
	* @param waitSemaphore (Optional) Semaphore that is waited on before the image is presented (only used if)
	*
	* @return vk::Result of the queue presentation
	*/
	vk::Result queuePresent(vk::Queue queue, uint32_t imageIndex, vk::Semaphore waitSemaphore)
	{
		vk::PresentInfoKHR presentInfo = {};
		presentInfo.swapchainCount = 1;
		presentInfo.pSwapchains = &swapChain;
		presentInfo.pImageIndices = &imageIndex;
		// Check if a wait semaphore has been specified to wait for before presenting the image
		if (waitSemaphore)
		{
			presentInfo.pWaitSemaphores = &waitSemaphore;
			presentInfo.waitSemaphoreCount = 1;
		}
		return queue.presentKHR(presentInfo);
	}


	/**
	* Destroy and free Vulkan resources used for the swapchain
	*/
	void cleanup()
	{
		if (swapChain)
		{
			for (uint32_t i = 0; i < imageCount; i++)
			{
				device.destroyImageView(buffers[i].view);
			}
		}
		if (surface)
		{
			device.destroySwapchainKHR(swapChain);
			instance.destroySurfaceKHR(surface);
		}
		surface = nullptr;
		swapChain = nullptr;
	}

#if defined(_DIRECT2DISPLAY)
	/**
	* Create direct to display surface
	*/	
	void createDirect2DisplaySurface(uint32_t width, uint32_t height)
	{
		uint32_t displayPropertyCount;
		
		// Get display property
		vkGetPhysicalDeviceDisplayPropertiesKHR(physicalDevice, &displayPropertyCount, NULL);
		vk::DisplayPropertiesKHR* pDisplayProperties = new vk::DisplayPropertiesKHR[displayPropertyCount];
		vkGetPhysicalDeviceDisplayPropertiesKHR(physicalDevice, &displayPropertyCount, pDisplayProperties);

		// Get plane property
		uint32_t planePropertyCount;
		vkGetPhysicalDeviceDisplayPlanePropertiesKHR(physicalDevice, &planePropertyCount, NULL);
		vk::DisplayPlanePropertiesKHR* pPlaneProperties = new vk::DisplayPlanePropertiesKHR[planePropertyCount];
		vkGetPhysicalDeviceDisplayPlanePropertiesKHR(physicalDevice, &planePropertyCount, pPlaneProperties);

		vk::DisplayKHR display;
		vk::DisplayModeKHR displayMode;
		vk::DisplayModePropertiesKHR* pModeProperties;
		bool foundMode = false;

		for(uint32_t i = 0; i < displayPropertyCount;++i)
		{
			display = pDisplayProperties[i].display;
			uint32_t modeCount;
			vkGetDisplayModePropertiesKHR(physicalDevice, display, &modeCount, NULL);
			pModeProperties = new vk::DisplayModePropertiesKHR[modeCount];
			vkGetDisplayModePropertiesKHR(physicalDevice, display, &modeCount, pModeProperties);

			for (uint32_t j = 0; j < modeCount; ++j)
			{
				const vk::DisplayModePropertiesKHR* mode = &pModeProperties[j];

				if (mode->parameters.visibleRegion.width == width && mode->parameters.visibleRegion.height == height)
				{
					displayMode = mode->displayMode;
					foundMode = true;
					break;
				}
			}
			if (foundMode)
			{
				break;
			}
			delete [] pModeProperties;
		}

		if(!foundMode)
		{
			vks::tools::exitFatal("Can't find a display and a display mode!", "Fatal error");
			return;
		}

		// Search for a best plane we can use
		uint32_t bestPlaneIndex = UINT32_MAX;
		vk::DisplayKHR* pDisplays = NULL;
		for(uint32_t i = 0; i < planePropertyCount; i++)
		{
			uint32_t planeIndex=i;
			uint32_t displayCount;
			vkGetDisplayPlaneSupportedDisplaysKHR(physicalDevice, planeIndex, &displayCount, NULL);
			if (pDisplays)
			{
				delete [] pDisplays;
			}
			pDisplays = new vk::DisplayKHR[displayCount];
			vkGetDisplayPlaneSupportedDisplaysKHR(physicalDevice, planeIndex, &displayCount, pDisplays);

			// Find a display that matches the current plane
			bestPlaneIndex = UINT32_MAX;
			for(uint32_t j = 0; j < displayCount; j++)
			{
				if(display == pDisplays[j])
				{
					bestPlaneIndex = i;
					break;
				}
			}
			if(bestPlaneIndex != UINT32_MAX)
			{
				break;
			}
		}

		if(bestPlaneIndex == UINT32_MAX)
		{
			vks::tools::exitFatal("Can't find a plane for displaying!", "Fatal error");
			return;
		}

		vk::DisplayPlaneCapabilitiesKHR planeCap;
		vkGetDisplayPlaneCapabilitiesKHR(physicalDevice, displayMode, bestPlaneIndex, &planeCap);
		vk::DisplayPlaneAlphaFlagBitsKHR alphaMode;

		if (planeCap.supportedAlpha & VK_DISPLAY_PLANE_ALPHA_PER_PIXEL_PREMULTIPLIED_BIT_KHR)
		{
			alphaMode = VK_DISPLAY_PLANE_ALPHA_PER_PIXEL_PREMULTIPLIED_BIT_KHR;
		}
		else if (planeCap.supportedAlpha & VK_DISPLAY_PLANE_ALPHA_PER_PIXEL_BIT_KHR)
		{

			alphaMode = VK_DISPLAY_PLANE_ALPHA_PER_PIXEL_BIT_KHR;
		}
		else
		{
			alphaMode = VK_DISPLAY_PLANE_ALPHA_GLOBAL_BIT_KHR;
		}

		vk::DisplaySurfaceCreateInfoKHR surfaceInfo{};
		//surfaceInfo.flags = 0;
		surfaceInfo.displayMode = displayMode;
		surfaceInfo.planeIndex = bestPlaneIndex;
		surfaceInfo.planeStackIndex = pPlaneProperties[bestPlaneIndex].currentStackIndex;
		surfaceInfo.transform = vk::SurfaceTransformFlagBitsKHR::eIdentity;
		surfaceInfo.globalAlpha = 1.0;
		surfaceInfo.alphaMode = alphaMode;
		surfaceInfo.imageExtent.width = width;
		surfaceInfo.imageExtent.height = height;

		vk::Result result = vkCreateDisplayPlaneSurfaceKHR(instance, &surfaceInfo, NULL, &surface);
		if(result !=vk::Result::eSuccess)
		{
			vks::tools::exitFatal("Failed to create surface!", "Fatal error");
		}

		delete[] pDisplays;
		delete[] pModeProperties;
		delete[] pDisplayProperties;
		delete[] pPlaneProperties;
	}
#endif 
};
