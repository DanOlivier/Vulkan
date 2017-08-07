/*
* Vulkan Example - Sparse texture residency example
*
* Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

/*
todos: 
- check sparse binding support on queue
- residencyNonResidentStrict
- meta data
- Run-time image data upload
*/

#include "vulkanexamplebase.h"
#include "VulkanTexture.hpp"
#include "VulkanHeightmap.hpp"
#include "keycodes.hpp"

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <assert.h>
#include <random>
#include <chrono>
#include <iostream>
#include <sstream>
#include <iomanip>

#define VERTEX_BUFFER_BIND_ID 0
#define ENABLE_VALIDATION false

// Vertex layout for this example
struct Vertex {
	float pos[3];
	float normal[3];
	float uv[2];
};

// Virtual texture page as a part of the partially resident texture
// Contains memory bindings, offsets and status information
struct VirtualTexturePage
{	
	vk::Offset3D offset;
	vk::Extent3D extent;
	vk::SparseImageMemoryBind imageMemoryBind;							// Sparse image memory bind for this page
	vk::DeviceSize size;													// Page (memory) size in bytes
	uint32_t mipLevel;													// Mip level that this page belongs to
	uint32_t layer;														// Array layer that this page belongs to
	uint32_t index;	

	VirtualTexturePage()
	{
		imageMemoryBind.memory = nullptr;						// Page initially not backed up by memory
	}

	// Allocate Vulkan memory for the virtual page
	void allocate(vk::Device device, uint32_t memoryTypeIndex)
	{
		if (imageMemoryBind.memory)
		{
			//std::cout << "Page " << index << " already allocated" << std::endl;
			return;
		};

		imageMemoryBind = vk::SparseImageMemoryBind();

		vk::MemoryAllocateInfo allocInfo = vks::initializers::memoryAllocateInfo();
		allocInfo.allocationSize = size;
		allocInfo.memoryTypeIndex = memoryTypeIndex;
		imageMemoryBind.memory = device.allocateMemory(allocInfo);

		vk::ImageSubresource subResource{};
		subResource.aspectMask = vk::ImageAspectFlagBits::eColor;
		subResource.mipLevel = mipLevel;
		subResource.arrayLayer = layer;

		// Sparse image memory binding
		imageMemoryBind.subresource = subResource;
		imageMemoryBind.extent = extent;
		imageMemoryBind.offset = offset;
	}

	// Release Vulkan memory allocated for this page
	void release(vk::Device device)
	{
		if (imageMemoryBind.memory)
		{
			device.freeMemory(imageMemoryBind.memory);
			imageMemoryBind.memory = nullptr;
			//std::cout << "Page " << index << " released" << std::endl;
		}
	}
};

// Virtual texture object containing all pages 
struct VirtualTexture
{
	vk::Device device;
	vk::Image image;														// Texture image handle
	vk::BindSparseInfo bindSparseInfo;									// Sparse queue binding information
	std::vector<VirtualTexturePage> pages;								// Contains all virtual pages of the texture
	std::vector<vk::SparseImageMemoryBind> sparseImageMemoryBinds;		// Sparse image memory bindings of all memory-backed virtual tables
	std::vector<vk::SparseMemoryBind>	opaqueMemoryBinds;					// Sparse ópaque memory bindings for the mip tail (if present)
	vk::SparseImageMemoryBindInfo imageMemoryBindInfo;					// Sparse image memory bind info 
	vk::SparseImageOpaqueMemoryBindInfo opaqueMemoryBindInfo;				// Sparse image opaque memory bind info (mip tail)
	uint32_t mipTailStart;												// First mip level in mip tail
	
	VirtualTexturePage* addPage(vk::Offset3D offset, vk::Extent3D extent, const vk::DeviceSize size, const uint32_t mipLevel, uint32_t layer)
	{
		VirtualTexturePage newPage;
		newPage.offset = offset;
		newPage.extent = extent;
		newPage.size = size;
		newPage.mipLevel = mipLevel;
		newPage.layer = layer;
		newPage.index = static_cast<uint32_t>(pages.size());
		newPage.imageMemoryBind.offset = offset;
		newPage.imageMemoryBind.extent = extent;
		pages.push_back(newPage);
		return &pages.back();
	}

	// Call before sparse binding to update memory bind list etc.
	void updateSparseBindInfo()
	{
		// Update list of memory-backed sparse image memory binds
		sparseImageMemoryBinds.resize(pages.size());
		uint32_t index = 0;
		for (auto page : pages)
		{
			sparseImageMemoryBinds[index] = page.imageMemoryBind;
			index++;
		}
		// Update sparse bind info
		bindSparseInfo = vks::initializers::bindSparseInfo();
		// todo: Semaphore for queue submission
		// bindSparseInfo.signalSemaphoreCount = 1;
		// bindSparseInfo.pSignalSemaphores = &bindSparseSemaphore;

		// Image memory binds
		imageMemoryBindInfo.image = image;
		imageMemoryBindInfo.bindCount = static_cast<uint32_t>(sparseImageMemoryBinds.size());
		imageMemoryBindInfo.pBinds = sparseImageMemoryBinds.data();
		bindSparseInfo.imageBindCount = (imageMemoryBindInfo.bindCount > 0) ? 1 : 0;
		bindSparseInfo.pImageBinds = &imageMemoryBindInfo;

		// Opaque image memory binds (mip tail)
		opaqueMemoryBindInfo.image = image;
		opaqueMemoryBindInfo.bindCount = static_cast<uint32_t>(opaqueMemoryBinds.size());
		opaqueMemoryBindInfo.pBinds = opaqueMemoryBinds.data();
		bindSparseInfo.imageOpaqueBindCount = (opaqueMemoryBindInfo.bindCount > 0) ? 1 : 0;
		bindSparseInfo.pImageOpaqueBinds = &opaqueMemoryBindInfo;
	}

	// Release all Vulkan resources
	void destroy()
	{
		for (auto page : pages)
		{
			page.release(device);
		}
		for (auto bind : opaqueMemoryBinds)
		{
			device.freeMemory(bind.memory);
		}
	}
};

uint32_t memoryTypeIndex;
int32_t lastFilledMip = 0;

class VulkanExample : public VulkanExampleBase
{
public:
	//todo: comments
	struct SparseTexture : VirtualTexture {
		vk::Sampler sampler;
		vk::ImageLayout imageLayout;
		vk::ImageView view;
		vk::DescriptorImageInfo descriptor;
		vk::Format format;
		uint32_t width, height;
		uint32_t mipLevels;
		uint32_t layerCount;
	} texture;

	struct {
		vks::Texture2D source;
	} textures;

	vks::HeightMap *heightMap = nullptr;

	struct {
		vk::PipelineVertexInputStateCreateInfo inputState;
		std::vector<vk::VertexInputBindingDescription> bindingDescriptions;
		std::vector<vk::VertexInputAttributeDescription> attributeDescriptions;
	} vertices;

	uint32_t indexCount;

	vks::Buffer uniformBufferVS;

	struct UboVS {
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

	//todo: comment
	vk::Semaphore bindSparseSemaphore;

	VulkanExample() : VulkanExampleBase(ENABLE_VALIDATION)
	{
		zoom = -1.3f; 
		rotation = { 76.25f, 0.0f, 0.0f }; 
		title = "Vulkan Example - Sparse texture residency";
		enableTextOverlay = true;
		std::cout.imbue(std::locale(""));
		camera.type = Camera::CameraType::firstperson;
		camera.movementSpeed = 50.0f;
#ifndef __ANDROID__
		camera.rotationSpeed = 0.25f;
#endif
		camera.position = { 84.5f, 40.5f, 225.0f };
		camera.setRotation(glm::vec3(-8.5f, -200.0f, 0.0f));
		camera.setPerspective(60.0f, (float)width / (float)height, 0.1f, 1024.0f);
		// Device features to be enabled for this example 
		enabledFeatures.shaderResourceResidency = VK_TRUE;
		enabledFeatures.shaderResourceMinLod = VK_TRUE;
	}

	~VulkanExample()
	{
		// Clean up used Vulkan resources 
		// Note : Inherited destructor cleans up resources stored in base class

		if (heightMap)
			delete heightMap;

		destroyTextureImage(texture);

		device.destroySemaphore(bindSparseSemaphore);

		device.destroyPipeline(pipelines.solid);

		device.destroyPipelineLayout(pipelineLayout);
		device.destroyDescriptorSetLayout(descriptorSetLayout);

		uniformBufferVS.destroy();
	}

    virtual void getEnabledFeatures()
    {
        if (deviceFeatures.sparseBinding && deviceFeatures.sparseResidencyImage2D) {
            enabledFeatures.sparseBinding = VK_TRUE;
            enabledFeatures.sparseResidencyImage2D = VK_TRUE;
        }
        else {
            std::cout << "Sparse binding not supported" << std::endl;
        }
    }

	glm::uvec3 alignedDivision(const vk::Extent3D& extent, const vk::Extent3D& granularity)
	{
		glm::uvec3 res;
		res.x = extent.width / granularity.width + ((extent.width  % granularity.width) ? 1u : 0u);
		res.y = extent.height / granularity.height + ((extent.height % granularity.height) ? 1u : 0u);
		res.z = extent.depth / granularity.depth + ((extent.depth  % granularity.depth) ? 1u : 0u);
		return res;
	}

	void prepareSparseTexture(uint32_t width, uint32_t height, uint32_t layerCount, vk::Format format)
	{
		texture.device = vulkanDevice->logicalDevice;
		texture.width = width;
		texture.height = height;
		texture.mipLevels = floor(log2(std::max(width, height))) + 1; 
		texture.layerCount = layerCount;
		texture.format = format;

		// Get device properites for the requested texture format
		vk::FormatProperties formatProperties;
		formatProperties = physicalDevice.getFormatProperties(format);

		// Get sparse image properties
		std::vector<vk::SparseImageFormatProperties> sparseProperties;
		// Get actual image format properties
		sparseProperties = physicalDevice.getSparseImageFormatProperties(
			format,
			vk::ImageType::e2D,
			vk::SampleCountFlagBits::e1,
			vk::ImageUsageFlagBits::eSampled,
			vk::ImageTiling::eOptimal);
		// Check if sparse is supported for this format
		uint32_t sparsePropertiesCount = sparseProperties.size();
		if (sparsePropertiesCount == 0)
		{
			std::cout << "Error: Requested format does not support sparse features!" << std::endl;
			return;
		}

		std::cout << "Sparse image format properties: " << sparsePropertiesCount << std::endl;
		for (auto props : sparseProperties)
		{
			std::cout << "\t Image granularity: w = " << props.imageGranularity.width << " h = " << props.imageGranularity.height << " d = " << props.imageGranularity.depth << std::endl;
			std::cout << "\t Aspect mask: " << vk::to_string(props.aspectMask) << std::endl;
			std::cout << "\t Flags: " << vk::to_string(props.flags) << std::endl;
		}

		// Create sparse image
		vk::ImageCreateInfo sparseImageCreateInfo = vks::initializers::imageCreateInfo();
		sparseImageCreateInfo.imageType = vk::ImageType::e2D;
		sparseImageCreateInfo.format = texture.format;
		sparseImageCreateInfo.mipLevels = texture.mipLevels;
		sparseImageCreateInfo.arrayLayers = texture.layerCount;
		sparseImageCreateInfo.samples = vk::SampleCountFlagBits::e1;
		sparseImageCreateInfo.tiling = vk::ImageTiling::eOptimal;
		sparseImageCreateInfo.usage = vk::ImageUsageFlagBits::eSampled;
		sparseImageCreateInfo.sharingMode = vk::SharingMode::eExclusive;
		sparseImageCreateInfo.initialLayout = vk::ImageLayout::eUndefined;
		sparseImageCreateInfo.extent = vk::Extent3D{ texture.width, texture.height, 1 };
		sparseImageCreateInfo.usage = vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled;
		sparseImageCreateInfo.flags = vk::ImageCreateFlagBits::eSparseBinding | vk::ImageCreateFlagBits::eSparseResidency;
		texture.image = device.createImage(sparseImageCreateInfo);

		// Get memory requirements
		vk::MemoryRequirements sparseImageMemoryReqs;
		// Sparse image memory requirement counts
		sparseImageMemoryReqs = device.getImageMemoryRequirements(texture.image);

		std::cout << "Image memory requirements:" << std::endl;
		std::cout << "\t Size: " << sparseImageMemoryReqs.size << std::endl;
		std::cout << "\t Alignment: " << sparseImageMemoryReqs.alignment << std::endl;

		// Check requested image size against hardware sparse limit
		if (sparseImageMemoryReqs.size > vulkanDevice->properties.limits.sparseAddressSpaceSize)
		{
			std::cout << "Error: Requested sparse image size exceeds supportes sparse address space size!" << std::endl;
			return;
		};

		// Get sparse memory requirements
		std::vector<vk::SparseImageMemoryRequirements> sparseMemoryReqs = 
			device.getImageSparseMemoryRequirements(texture.image);
		uint32_t sparseMemoryReqsCount = sparseMemoryReqs.size();
		if (sparseMemoryReqsCount == 0)
		{
			std::cout << "Error: No memory requirements for the sparse image!" << std::endl;
			return;
		}

		std::cout << "Sparse image memory requirements: " << sparseMemoryReqsCount << std::endl;
		for (auto reqs : sparseMemoryReqs)
		{
			std::cout << "\t Image granularity: w = " << reqs.formatProperties.imageGranularity.width << " h = " << reqs.formatProperties.imageGranularity.height << " d = " << reqs.formatProperties.imageGranularity.depth << std::endl;
			std::cout << "\t Mip tail first LOD: " << reqs.imageMipTailFirstLod << std::endl;
			std::cout << "\t Mip tail size: " << reqs.imageMipTailSize << std::endl;
			std::cout << "\t Mip tail offset: " << reqs.imageMipTailOffset << std::endl;
			std::cout << "\t Mip tail stride: " << reqs.imageMipTailStride << std::endl;
			//todo:multiple reqs
			texture.mipTailStart = reqs.imageMipTailFirstLod;
		}
		
		lastFilledMip = texture.mipTailStart - 1;

		// Get sparse image requirements for the color aspect
		vk::SparseImageMemoryRequirements sparseMemoryReq;
		bool colorAspectFound = false;
		for (auto reqs : sparseMemoryReqs)
		{
			if (reqs.formatProperties.aspectMask & vk::ImageAspectFlagBits::eColor)
			{
				sparseMemoryReq = reqs;
				colorAspectFound = true;
				break;
			}
		}
		if (!colorAspectFound)
		{
			std::cout << "Error: Could not find sparse image memory requirements for color aspect bit!" << std::endl;
			return;
		}

		// todo:
		// Calculate number of required sparse memory bindings by alignment
		assert((sparseImageMemoryReqs.size % sparseImageMemoryReqs.alignment) == 0);
		memoryTypeIndex = vulkanDevice->getMemoryType(sparseImageMemoryReqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);

		// Get sparse bindings
		uint32_t sparseBindsCount = static_cast<uint32_t>(sparseImageMemoryReqs.size / sparseImageMemoryReqs.alignment);		
		std::vector<vk::SparseMemoryBind>	sparseMemoryBinds(sparseBindsCount);

		// Check if the format has a single mip tail for all layers or one mip tail for each layer
		// The mip tail contains all mip levels > sparseMemoryReq.imageMipTailFirstLod
		bool singleMipTail = !!(sparseMemoryReq.formatProperties.flags & vk::SparseImageFormatFlagBits::eSingleMiptail);

		// Sparse bindings for each mip level of all layers outside of the mip tail
		for (uint32_t layer = 0; layer < texture.layerCount; layer++)
		{
			// sparseMemoryReq.imageMipTailFirstLod is the first mip level that's stored inside the mip tail
			for (uint32_t mipLevel = 0; mipLevel < sparseMemoryReq.imageMipTailFirstLod; mipLevel++)
			{
				vk::Extent3D extent;
				extent.width = std::max(sparseImageCreateInfo.extent.width >> mipLevel, 1u);
				extent.height = std::max(sparseImageCreateInfo.extent.height >> mipLevel, 1u);
				extent.depth = std::max(sparseImageCreateInfo.extent.depth >> mipLevel, 1u);

				vk::ImageSubresource subResource{};
				subResource.aspectMask = vk::ImageAspectFlagBits::eColor;
				subResource.mipLevel = mipLevel;
				subResource.arrayLayer = layer;

				// Aligned sizes by image granularity
				vk::Extent3D imageGranularity = sparseMemoryReq.formatProperties.imageGranularity;
				glm::uvec3 sparseBindCounts = alignedDivision(extent, imageGranularity);
				glm::uvec3 lastBlockExtent;
				lastBlockExtent.x = (extent.width % imageGranularity.width) ? extent.width % imageGranularity.width : imageGranularity.width;
				lastBlockExtent.y = (extent.height % imageGranularity.height) ? extent.height % imageGranularity.height : imageGranularity.height;
				lastBlockExtent.z = (extent.depth % imageGranularity.depth) ? extent.depth % imageGranularity.depth : imageGranularity.depth;

				// Alllocate memory for some blocks
				uint32_t index = 0;
				for (uint32_t z = 0; z < sparseBindCounts.z; z++)
				{
					for (uint32_t y = 0; y < sparseBindCounts.y; y++)
					{
						for (uint32_t x = 0; x < sparseBindCounts.x; x++)
						{
							// Offset 
							vk::Offset3D offset;
							offset.x = x * imageGranularity.width;
							offset.y = y * imageGranularity.height;
							offset.z = z * imageGranularity.depth;
							// Size of the page
							vk::Extent3D extent;
							extent.width = (x == sparseBindCounts.x - 1) ? lastBlockExtent.x : imageGranularity.width;
							extent.height = (y == sparseBindCounts.y - 1) ? lastBlockExtent.y : imageGranularity.height;
							extent.depth = (z == sparseBindCounts.z - 1) ? lastBlockExtent.z : imageGranularity.depth;

							// Add new virtual page
							VirtualTexturePage *newPage = texture.addPage(offset, extent, sparseImageMemoryReqs.alignment, mipLevel, layer);
							newPage->imageMemoryBind.subresource = subResource;

							if ((x % 2 == 1) || (y % 2 == 1))
							{
								// Allocate memory for this virtual page
								//newPage->allocate(device, memoryTypeIndex);
							}

							index++;
						}
					}
				}
			}

			// Check if format has one mip tail per layer
			if ((!singleMipTail) && (sparseMemoryReq.imageMipTailFirstLod < texture.mipLevels))
			{
				// Allocate memory for the mip tail
				vk::MemoryAllocateInfo allocInfo = vks::initializers::memoryAllocateInfo();
				allocInfo.allocationSize = sparseMemoryReq.imageMipTailSize;
				allocInfo.memoryTypeIndex = memoryTypeIndex;

				vk::DeviceMemory deviceMemory;
				deviceMemory = device.allocateMemory(allocInfo);

				// (Opaque) sparse memory binding
				vk::SparseMemoryBind sparseMemoryBind{};
				sparseMemoryBind.resourceOffset = sparseMemoryReq.imageMipTailOffset + layer * sparseMemoryReq.imageMipTailStride;
				sparseMemoryBind.size = sparseMemoryReq.imageMipTailSize;
				sparseMemoryBind.memory = deviceMemory;

				texture.opaqueMemoryBinds.push_back(sparseMemoryBind);
			}
		} // end layers and mips

		std::cout << "Texture info:" << std::endl;
		std::cout << "\tDim: " << texture.width << " x " << texture.height << std::endl;
		std::cout << "\tVirtual pages: " << texture.pages.size() << std::endl;

		// Check if format has one mip tail for all layers
		if ((sparseMemoryReq.formatProperties.flags & vk::SparseImageFormatFlagBits::eSingleMiptail) && (sparseMemoryReq.imageMipTailFirstLod < texture.mipLevels))
		{
			// Allocate memory for the mip tail
			vk::MemoryAllocateInfo allocInfo = vks::initializers::memoryAllocateInfo();
			allocInfo.allocationSize = sparseMemoryReq.imageMipTailSize;
			allocInfo.memoryTypeIndex = memoryTypeIndex;

			vk::DeviceMemory deviceMemory;
			deviceMemory = device.allocateMemory(allocInfo);

			// (Opaque) sparse memory binding
			vk::SparseMemoryBind sparseMemoryBind{};
			sparseMemoryBind.resourceOffset = sparseMemoryReq.imageMipTailOffset;
			sparseMemoryBind.size = sparseMemoryReq.imageMipTailSize;
			sparseMemoryBind.memory = deviceMemory;

			texture.opaqueMemoryBinds.push_back(sparseMemoryBind);
		}

		// Create signal semaphore for sparse binding
		vk::SemaphoreCreateInfo semaphoreCreateInfo = vks::initializers::semaphoreCreateInfo();
		bindSparseSemaphore = device.createSemaphore(semaphoreCreateInfo);

		// Prepare bind sparse info for reuse in queue submission
		texture.updateSparseBindInfo();

		// Bind to queue
		// todo: in draw?
		queue.bindSparse(texture.bindSparseInfo, vk::Fence(nullptr));
		//todo: use sparse bind semaphore
		queue.waitIdle();

		// Create sampler
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
		sampler.maxLod = static_cast<float>(texture.mipLevels);
		sampler.anisotropyEnable = vulkanDevice->features.samplerAnisotropy;
		sampler.maxAnisotropy = vulkanDevice->features.samplerAnisotropy ? vulkanDevice->properties.limits.maxSamplerAnisotropy : 1.0f;
		sampler.anisotropyEnable = false;
		sampler.borderColor = vk::BorderColor::eFloatOpaqueWhite;
		texture.sampler = device.createSampler(sampler);

		// Create image view
		vk::ImageViewCreateInfo view = vks::initializers::imageViewCreateInfo();
		view.viewType = vk::ImageViewType::e2D;
		view.format = format;
		view.components = { vk::ComponentSwizzle::eR, vk::ComponentSwizzle::eG, vk::ComponentSwizzle::eB, vk::ComponentSwizzle::eA };
		view.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
		view.subresourceRange.baseMipLevel = 0;
		view.subresourceRange.baseArrayLayer = 0;
		view.subresourceRange.layerCount = 1;
		view.subresourceRange.levelCount = texture.mipLevels;
		view.image = texture.image;
		texture.view = device.createImageView(view);

		// Fill image descriptor image info that can be used during the descriptor set setup
		texture.descriptor.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
		texture.descriptor.imageView = texture.view;
		texture.descriptor.sampler = texture.sampler;

		// Fill smallest (non-tail) mip map leve
		fillVirtualTexture(lastFilledMip);
	}

	// Free all Vulkan resources used a texture object
	void destroyTextureImage(SparseTexture texture)
	{
		device.destroyImageView(texture.view);
		device.destroyImage(texture.image);
		device.destroySampler(texture.sampler);
		texture.destroy();
	}

	void buildCommandBuffers()
	{
		vk::CommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();

		vk::ClearValue clearValues[2];
		clearValues[0].color = vk::ClearColorValue{ std::array<float, 4>{ 0.0f, 0.0f, 0.2f, 1.0f } };
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
			drawCmdBuffers[i].bindVertexBuffers(VERTEX_BUFFER_BIND_ID, heightMap->vertexBuffer.buffer, offsets);
			drawCmdBuffers[i].bindIndexBuffer(heightMap->indexBuffer.buffer, 0, vk::IndexType::eUint32);
			drawCmdBuffers[i].drawIndexed(heightMap->indexCount, 1, 0, 0, 0);

			drawCmdBuffers[i].endRenderPass();

			drawCmdBuffers[i].end();
		}
	}

	void draw()
	{
		VulkanExampleBase::prepareFrame();

		// Sparse bindings
//		queue.bindSparse(bindSparseInfo, vk::Fence(nullptr));
		//todo: use sparse bind semaphore
//		queue.waitIdle();

		// Command buffer to be sumitted to the queue
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &drawCmdBuffers[currentBuffer];

		// Submit to queue
		queue.submit(submitInfo, vk::Fence(nullptr));

		VulkanExampleBase::submitFrame();
	}

	void loadAssets()
	{
		textures.source.loadFromFile(getAssetPath() + "textures/ground_dry_bc3_unorm.ktx", vk::Format::eBc3UnormBlock, vulkanDevice, queue, vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eSampled);
	}

	// Generate a terrain quad patch for feeding to the tessellation control shader
	void generateTerrain()
	{
		heightMap = new vks::HeightMap(vulkanDevice, queue);
#if defined(__ANDROID__)
		heightMap->loadFromFile(getAssetPath() + "textures/terrain_heightmap_r16.ktx", 128, glm::vec3(2.0f, 48.0f, 2.0f), vks::HeightMap::topologyTriangles, androidApp->activity->assetManager);
#else
		heightMap->loadFromFile(getAssetPath() + "textures/terrain_heightmap_r16.ktx", 128, glm::vec3(2.0f, 48.0f, 2.0f), vks::HeightMap::topologyTriangles);
#endif
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
		// Location 1 : Vertex normal
		vertices.attributeDescriptions[1] =
			vks::initializers::vertexInputAttributeDescription(
				VERTEX_BUFFER_BIND_ID,
				1,
				vk::Format::eR32G32B32Sfloat,
				offsetof(Vertex, normal));
		// Location 1 : Texture coordinates
		vertices.attributeDescriptions[2] =
			vks::initializers::vertexInputAttributeDescription(
				VERTEX_BUFFER_BIND_ID,
				2,
				vk::Format::eR32G32Sfloat,
				offsetof(Vertex, uv));

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

		std::vector<vk::WriteDescriptorSet> writeDescriptorSets =
		{
			// Binding 0 : Vertex shader uniform buffer
			vks::initializers::writeDescriptorSet(
				descriptorSet, 
				vk::DescriptorType::eUniformBuffer, 
				0, 
				&uniformBufferVS.descriptor),
			// Binding 1 : Fragment shader texture sampler
			vks::initializers::writeDescriptorSet(
				descriptorSet, 
				vk::DescriptorType::eCombinedImageSampler, 
				1, 
				&texture.descriptor)
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
			vks::initializers::pipelineMultisampleStateCreateInfo(
				vk::SampleCountFlagBits::e1);

		std::vector<vk::DynamicState> dynamicStateEnables = {
			vk::DynamicState::eViewport,
			vk::DynamicState::eScissor
		};
		vk::PipelineDynamicStateCreateInfo dynamicState =
			vks::initializers::pipelineDynamicStateCreateInfo(dynamicStateEnables);

		// Load shaders
		std::array<vk::PipelineShaderStageCreateInfo,2> shaderStages;

		shaderStages[0] = loadShader(getAssetPath() + "shaders/texturesparseresidency/sparseresidency.vert.spv", vk::ShaderStageFlagBits::eVertex);
		shaderStages[1] = loadShader(getAssetPath() + "shaders/texturesparseresidency/sparseresidency.frag.spv", vk::ShaderStageFlagBits::eFragment);

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

		uboVS.projection = camera.matrices.perspective;
		uboVS.model = camera.matrices.view;
		//uboVS.model = glm::mat4();

		uboVS.viewPos = glm::vec4(0.0f, 0.0f, -zoom, 0.0f);

		uniformBufferVS.map();
		memcpy(uniformBufferVS.mapped, &uboVS, sizeof(uboVS));
		uniformBufferVS.unmap();
	}

	void prepare()
	{
		VulkanExampleBase::prepare();
		// Check if the GPU supports sparse residency for 2D images
		if (!vulkanDevice->features.sparseResidencyImage2D)
		{
			vks::tools::exitFatal("Device does not support sparse residency for 2D images!", "Feature not supported");
		}
		loadAssets();
		generateTerrain();
		setupVertexDescriptions();
		prepareUniformBuffers();
		// Create a virtual texture with max. possible dimension (does not take up any VRAM yet)
		prepareSparseTexture(8192, 8192, 1, vk::Format::eR8G8B8A8Unorm);
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

	// Clear all pages of the virtual texture
	// todo: just for testing
	void flushVirtualTexture()
	{
		vkDeviceWaitIdle(device);
		for (auto& page : texture.pages)
		{
			page.release(device);
		}
		texture.updateSparseBindInfo();
		queue.bindSparse(texture.bindSparseInfo, vk::Fence(nullptr));
		//todo: use sparse bind semaphore
		queue.waitIdle();
		lastFilledMip = texture.mipTailStart - 1;
	}

	// Fill a complete mip level
	void fillVirtualTexture(int32_t &mipLevel)
	{
		vkDeviceWaitIdle(device);
		std::default_random_engine rndEngine(std::random_device{}());
		std::uniform_real_distribution<float> rndDist(0.0f, 1.0f);
		std::vector<vk::ImageBlit> imageBlits;
		for (auto& page : texture.pages)
		{
			if ((page.mipLevel == mipLevel) && /*(rndDist(rndEngine) < 0.5f) &&*/ !page.imageMemoryBind.memory)
			{
				// Allocate page memory
				page.allocate(device, memoryTypeIndex);

				// Current mip level scaling
				uint32_t scale = texture.width / (texture.width >> page.mipLevel);

				for (uint32_t x = 0; x < scale; x++)
				{
					for (uint32_t y = 0; y < scale; y++)
					{
						// Image blit
						vk::ImageBlit blit{};
						// Source
						blit.srcSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
						blit.srcSubresource.baseArrayLayer = 0;
						blit.srcSubresource.layerCount = 1;
						blit.srcSubresource.mipLevel = 0;
						blit.srcOffsets[0] = vk::Offset3D{ 0, 0, 0 };
						blit.srcOffsets[1] = vk::Offset3D{ static_cast<int32_t>(textures.source.width), static_cast<int32_t>(textures.source.height), 1 };
						// Dest
						blit.dstSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
						blit.dstSubresource.baseArrayLayer = 0;
						blit.dstSubresource.layerCount = 1;
						blit.dstSubresource.mipLevel = page.mipLevel;
						blit.dstOffsets[0].x = static_cast<int32_t>(page.offset.x + x * 128 / scale);
						blit.dstOffsets[0].y = static_cast<int32_t>(page.offset.y + y * 128 / scale);
						blit.dstOffsets[0].z = 0;
						blit.dstOffsets[1].x = static_cast<int32_t>(blit.dstOffsets[0].x + page.extent.width / scale);
						blit.dstOffsets[1].y = static_cast<int32_t>(blit.dstOffsets[0].y + page.extent.height / scale);
						blit.dstOffsets[1].z = 1;

						imageBlits.push_back(blit);
					}
				}
			}
		}

		// Update sparse queue binding
		texture.updateSparseBindInfo();
		queue.bindSparse(texture.bindSparseInfo, vk::Fence(nullptr));
		//todo: use sparse bind semaphore
		queue.waitIdle();

		// Issue blit commands
		if (imageBlits.size() > 0)
		{
			auto tStart = std::chrono::high_resolution_clock::now();

			vk::CommandBuffer copyCmd = vulkanDevice->createCommandBuffer(vk::CommandBufferLevel::ePrimary, true);

			copyCmd.blitImage(
				textures.source.image,
				vk::ImageLayout::eTransferSrcOptimal,
				texture.image,
				vk::ImageLayout::eTransferDstOptimal,
				imageBlits,
				vk::Filter::eLinear
			);

			vulkanDevice->flushCommandBuffer(copyCmd, queue);

			auto tEnd = std::chrono::high_resolution_clock::now();
			auto tDiff = std::chrono::duration<double, std::milli>(tEnd - tStart).count();
			std::cout << "Image blits took " << tDiff << " ms" << std::endl;
		}

		queue.waitIdle();

		mipLevel--;
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
		case KEY_F:
			flushVirtualTexture();
			break;
		case KEY_N:
			if (lastFilledMip >= 0)
			{
				fillVirtualTexture(lastFilledMip);
			}
			break;
		}
	}

	virtual void getOverlayText(VulkanTextOverlay *textOverlay)
	{
		uint32_t respages = 0;
		std::for_each(texture.pages.begin(), texture.pages.end(), [&respages](VirtualTexturePage page) { respages += (page.imageMemoryBind.memory) ? 1 :0; });
		std::stringstream ss;
		ss << std::setprecision(2) << std::fixed << uboVS.lodBias;
#if defined(__ANDROID__)
//		textOverlay->addText("LOD bias: " + ss.str() + " (Buttons L1/R1 to change)", 5.0f, 85.0f, VulkanTextOverlay::alignLeft);
#else
		//textOverlay->addText("LOD bias: " + ss.str() + " (numpad +/- to change)", 5.0f, 85.0f, VulkanTextOverlay::alignLeft);
		textOverlay->addText("Resident pages: " + std::to_string(respages) + " / " + std::to_string(texture.pages.size()), 5.0f, 85.0f, VulkanTextOverlay::alignLeft);
		textOverlay->addText("\"n\" to fill next mip level (" + std::to_string(lastFilledMip) + ")", 5.0f, 100.0f, VulkanTextOverlay::alignLeft);
#endif
	}
};

VULKAN_EXAMPLE_MAIN()
