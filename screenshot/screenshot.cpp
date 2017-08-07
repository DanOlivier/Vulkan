/*
* Vulkan Example - Taking screenshots
*
* Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include "vulkanexamplebase.h"
#include "VulkanModel.hpp"

#include <fstream>

#define VERTEX_BUFFER_BIND_ID 0
#define ENABLE_VALIDATION false

class VulkanExample : public VulkanExampleBase
{
public:
	// Vertex layout for the models
	vks::VertexLayout vertexLayout = vks::VertexLayout({
		vks::VERTEX_COMPONENT_POSITION,
		vks::VERTEX_COMPONENT_NORMAL,
		vks::VERTEX_COMPONENT_COLOR,
	});

	struct {
		vks::Model object;
	} models;

	vks::Buffer uniformBuffer;

	struct {
		glm::mat4 projection;
		glm::mat4 model;
		glm::mat4 view;
		int32_t texIndex = 0;
	} uboVS;

	vk::PipelineLayout pipelineLayout;
	vk::Pipeline pipeline;
	vk::DescriptorSetLayout descriptorSetLayout;
	vk::DescriptorSet descriptorSet;

	VulkanExample() : VulkanExampleBase(ENABLE_VALIDATION)
	{
		title = "Vulkan Example - Screenshot";
		enableTextOverlay = true;

		camera.type = Camera::CameraType::lookat;
		camera.setPerspective(60.0f, (float)width / (float)height, 0.1f, 512.0f);
		camera.setRotation(glm::vec3(-25.0f, 23.75f, 0.0f));
		camera.setTranslation(glm::vec3(0.0f, 0.0f, -2.0f));
	}

	~VulkanExample()
	{
		// Clean up used Vulkan resources 
		// Note : Inherited destructor cleans up resources stored in base class
		device.destroyPipeline(pipeline);
		device.destroyPipelineLayout(pipelineLayout);
		device.destroyDescriptorSetLayout(descriptorSetLayout);

		models.object.destroy();

		uniformBuffer.destroy();
	}

	void loadAssets()
	{
		models.object.loadFromFile(getAssetPath() + "models/chinesedragon.dae", vertexLayout, 0.1f, vulkanDevice, queue);
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

			vk::Rect2D scissor = vks::initializers::rect2D(width, height,	0, 0);
			drawCmdBuffers[i].setScissor(0, scissor);

			drawCmdBuffers[i].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSet, nullptr);
			drawCmdBuffers[i].bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);

			std::vector<vk::DeviceSize> offsets = { 0 };
			drawCmdBuffers[i].bindVertexBuffers(VERTEX_BUFFER_BIND_ID, models.object.vertices.buffer, offsets);
			drawCmdBuffers[i].bindIndexBuffer(models.object.indices.buffer, 0, vk::IndexType::eUint32);

			drawCmdBuffers[i].drawIndexed(models.object.indexCount, 1, 0, 0, 0);

			drawCmdBuffers[i].endRenderPass();

			drawCmdBuffers[i].end();
		}
	}

	void setupDescriptorPool()
	{
		// Example uses one ubo and one image sampler
		std::vector<vk::DescriptorPoolSize> poolSizes = {
			vks::initializers::descriptorPoolSize(vk::DescriptorType::eUniformBuffer, 1),
		};

		vk::DescriptorPoolCreateInfo descriptorPoolInfo =
			vks::initializers::descriptorPoolCreateInfo(
				poolSizes.size(),
				poolSizes.data(),
				2);

		descriptorPool = device.createDescriptorPool(descriptorPoolInfo);
	}

	void setupDescriptorSetLayout()
	{
		std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings = {			
			vks::initializers::descriptorSetLayoutBinding(vk::DescriptorType::eUniformBuffer, vk::ShaderStageFlagBits::eVertex, 0),	// Binding 0 : Vertex shader uniform buffer
		};

		vk::DescriptorSetLayoutCreateInfo descriptorLayout =
			vks::initializers::descriptorSetLayoutCreateInfo(
				setLayoutBindings.data(),
				setLayoutBindings.size());

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

		std::vector<vk::WriteDescriptorSet> writeDescriptorSets = {			
			vks::initializers::writeDescriptorSet(descriptorSet, vk::DescriptorType::eUniformBuffer, 0, &uniformBuffer.descriptor),	// Binding 0 : Vertex shader uniform buffer
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

		vk::GraphicsPipelineCreateInfo pipelineCreateInfo =
			vks::initializers::pipelineCreateInfo(
				pipelineLayout,
				renderPass);

		std::array<vk::PipelineShaderStageCreateInfo, 2> shaderStages;

		pipelineCreateInfo.pInputAssemblyState = &inputAssemblyState;
		pipelineCreateInfo.pRasterizationState = &rasterizationState;
		pipelineCreateInfo.pColorBlendState = &colorBlendState;
		pipelineCreateInfo.pMultisampleState = &multisampleState;
		pipelineCreateInfo.pViewportState = &viewportState;
		pipelineCreateInfo.pDepthStencilState = &depthStencilState;
		pipelineCreateInfo.pDynamicState = &dynamicState;
		pipelineCreateInfo.stageCount = shaderStages.size();
		pipelineCreateInfo.pStages = shaderStages.data();

		// Vertex bindings and attributes
		// Binding description
		std::vector<vk::VertexInputBindingDescription> vertexInputBindings = {
			vks::initializers::vertexInputBindingDescription(0, vertexLayout.stride(), vk::VertexInputRate::eVertex),
		};

		// Attribute descriptions
		std::vector<vk::VertexInputAttributeDescription> vertexInputAttributes = {
			vks::initializers::vertexInputAttributeDescription(0, 0, vk::Format::eR32G32B32Sfloat, 0),					// Position
			vks::initializers::vertexInputAttributeDescription(0, 1, vk::Format::eR32G32B32Sfloat, sizeof(float) * 3),	// Normal
			vks::initializers::vertexInputAttributeDescription(0, 2, vk::Format::eR32G32B32Sfloat, sizeof(float) * 6),	// Color
		};

		vk::PipelineVertexInputStateCreateInfo vertexInputState = vks::initializers::pipelineVertexInputStateCreateInfo();
		vertexInputState.vertexBindingDescriptionCount = static_cast<uint32_t>(vertexInputBindings.size());
		vertexInputState.pVertexBindingDescriptions = vertexInputBindings.data();
		vertexInputState.vertexAttributeDescriptionCount = static_cast<uint32_t>(vertexInputAttributes.size());
		vertexInputState.pVertexAttributeDescriptions = vertexInputAttributes.data();

		pipelineCreateInfo.pVertexInputState = &vertexInputState;

		// Mesh rendering pipeline
		shaderStages[0] = loadShader(getAssetPath() + "shaders/screenshot/mesh.vert.spv", vk::ShaderStageFlagBits::eVertex);
		shaderStages[1] = loadShader(getAssetPath() + "shaders/screenshot/mesh.frag.spv", vk::ShaderStageFlagBits::eFragment);
		pipeline = device.createGraphicsPipelines(pipelineCache, pipelineCreateInfo)[0];
	}

	void prepareUniformBuffers()
	{
		// Vertex shader uniform buffer block
		vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eUniformBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			&uniformBuffer,
			sizeof(uboVS));

		updateUniformBuffers();
	}

	void updateUniformBuffers()
	{
		uboVS.projection = camera.matrices.perspective;
		uboVS.view = camera.matrices.view;
		uboVS.model = glm::mat4();

		uniformBuffer.map();
		uniformBuffer.copyTo(&uboVS, sizeof(uboVS));
		uniformBuffer.unmap();
	}

	// Take a screenshot for the curretn swapchain image
	// This is done using a blit from the swapchain image to a linear image whose memory content is then saved as a ppm image
	// Getting the image date directly from a swapchain image wouldn't work as they're usually stored in an implementation dependant optimal tiling format
	// Note: This requires the swapchain images to be created with the vk::ImageUsageFlagBits::eTransferSrc flag (see VulkanSwapChain::create)
	void saveScreenshot(const char *filename)
	{
		// Get format properties for the swapchain color format
		vk::FormatProperties formatProps;

		bool supportsBlit = true;

		// Check blit support for source and destination

		// Check if the device supports blitting from optimal images (the swapchain images are in optimal format)
		formatProps = physicalDevice.getFormatProperties(swapChain.colorFormat);
		if (!(formatProps.optimalTilingFeatures & vk::FormatFeatureFlagBits::eBlitSrc)) {
			std::cerr << "Device does not support blitting from optimal tiled images, using copy instead of blit!" << std::endl;
			supportsBlit = false;
		}

		// Check if the device supports blitting to linear images 
		formatProps = physicalDevice.getFormatProperties(vk::Format::eR8G8B8A8Unorm);
		if (!(formatProps.linearTilingFeatures & vk::FormatFeatureFlagBits::eBlitDst)) {
			std::cerr << "Device does not support blitting to linear tiled images, using copy instead of blit!" << std::endl;
			supportsBlit = false;
		}

		// Source for the copy is the last rendered swapchain image
		vk::Image srcImage = swapChain.images[currentBuffer];
	
		// Create the linear tiled destination image to copy to and to read the memory from
		vk::ImageCreateInfo imgCreateInfo(vks::initializers::imageCreateInfo());
		imgCreateInfo.imageType = vk::ImageType::e2D;
		// Note that vkCmdBlitImage (if supported) will also do format conversions if the swapchain color format would differ
		imgCreateInfo.format = vk::Format::eR8G8B8A8Unorm;
		imgCreateInfo.extent.width = width;
		imgCreateInfo.extent.height = height;
		imgCreateInfo.extent.depth = 1;
		imgCreateInfo.arrayLayers = 1;
		imgCreateInfo.mipLevels = 1;
		imgCreateInfo.initialLayout = vk::ImageLayout::eUndefined;
		imgCreateInfo.samples = vk::SampleCountFlagBits::e1;
		imgCreateInfo.tiling = vk::ImageTiling::eLinear;
		imgCreateInfo.usage = vk::ImageUsageFlagBits::eTransferDst;
		// Create the image
		vk::Image dstImage;
		dstImage = device.createImage(imgCreateInfo);
		// Create memory to back up the image
		vk::MemoryRequirements memRequirements;
		vk::MemoryAllocateInfo memAllocInfo(vks::initializers::memoryAllocateInfo());
		vk::DeviceMemory dstImageMemory;
		memRequirements = device.getImageMemoryRequirements(dstImage);
		memAllocInfo.allocationSize = memRequirements.size;
		// Memory must be host visible to copy from
		memAllocInfo.memoryTypeIndex = vulkanDevice->getMemoryType(memRequirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
		dstImageMemory = device.allocateMemory(memAllocInfo);
		device.bindImageMemory(dstImage, dstImageMemory, 0);

		// Do the actual blit from the swapchain image to our host visible destination image
		vk::CommandBuffer copyCmd = vulkanDevice->createCommandBuffer(vk::CommandBufferLevel::ePrimary, true);

		//vk::ImageMemoryBarrier imageMemoryBarrier = vks::initializers::imageMemoryBarrier();
		
		// Transition destination image to transfer destination layout
		vks::tools::insertImageMemoryBarrier(
			copyCmd,
			dstImage,
			vk::AccessFlags(),
			vk::AccessFlagBits::eTransferWrite,
			vk::ImageLayout::eUndefined,
			vk::ImageLayout::eTransferDstOptimal,
			vk::PipelineStageFlagBits::eTransfer,
			vk::PipelineStageFlagBits::eTransfer,
			vk::ImageSubresourceRange( vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 ));

		// Transition swapchain image from present to transfer source layout
		vks::tools::insertImageMemoryBarrier(
			copyCmd,
			srcImage,
			vk::AccessFlagBits::eMemoryRead,
			vk::AccessFlagBits::eTransferRead,
			vk::ImageLayout::ePresentSrcKHR,
			vk::ImageLayout::eTransferSrcOptimal,
			vk::PipelineStageFlagBits::eTransfer,
			vk::PipelineStageFlagBits::eTransfer,
			vk::ImageSubresourceRange( vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 ));

		// If source and destination support blit we'll blit as this also does automatic format conversion (e.g. from BGR to RGB)
		if (supportsBlit)
		{
			// Define the region to blit (we will blit the whole swapchain image)
			vk::Offset3D blitSize;
			blitSize.x = width;
			blitSize.y = height;
			blitSize.z = 1;
			vk::ImageBlit imageBlitRegion{};
			imageBlitRegion.srcSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
			imageBlitRegion.srcSubresource.layerCount = 1;
			imageBlitRegion.srcOffsets[1] = blitSize;
			imageBlitRegion.dstSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
			imageBlitRegion.dstSubresource.layerCount = 1;
			imageBlitRegion.dstOffsets[1] = blitSize;

			// Issue the blit command
			copyCmd.blitImage(
				srcImage, vk::ImageLayout::eTransferSrcOptimal,
				dstImage, vk::ImageLayout::eTransferDstOptimal,
				imageBlitRegion,
				vk::Filter::eNearest);
		}
		else
		{
			// Otherwise use image copy (requires us to manually flip components)
			vk::ImageCopy imageCopyRegion{};
			imageCopyRegion.srcSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
			imageCopyRegion.srcSubresource.layerCount = 1;
			imageCopyRegion.dstSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
			imageCopyRegion.dstSubresource.layerCount = 1;
			imageCopyRegion.extent.width = width;
			imageCopyRegion.extent.height = height;
			imageCopyRegion.extent.depth = 1;

			// Issue the copy command
			copyCmd.copyImage(
				srcImage, vk::ImageLayout::eTransferSrcOptimal,
				dstImage, vk::ImageLayout::eTransferDstOptimal,
				imageCopyRegion);
		}

		// Transition destination image to general layout, which is the required layout for mapping the image memory later on
		vks::tools::insertImageMemoryBarrier(
			copyCmd,
			dstImage,
			vk::AccessFlagBits::eTransferWrite,
			vk::AccessFlagBits::eMemoryRead,
			vk::ImageLayout::eTransferDstOptimal,
			vk::ImageLayout::eGeneral,
			vk::PipelineStageFlagBits::eTransfer,
			vk::PipelineStageFlagBits::eTransfer,
			vk::ImageSubresourceRange{ vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 });

		// Transition back the swap chain image after the blit is done
		vks::tools::insertImageMemoryBarrier(
			copyCmd,
			srcImage,
			vk::AccessFlagBits::eTransferRead,
			vk::AccessFlagBits::eMemoryRead,
			vk::ImageLayout::eTransferSrcOptimal,
			vk::ImageLayout::ePresentSrcKHR,
			vk::PipelineStageFlagBits::eTransfer,
			vk::PipelineStageFlagBits::eTransfer,
			vk::ImageSubresourceRange{ vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 });

		vulkanDevice->flushCommandBuffer(copyCmd, queue);

		// Get layout of the image (including row pitch)
		vk::ImageSubresource subResource{};
		subResource.aspectMask = vk::ImageAspectFlagBits::eColor;
		vk::SubresourceLayout subResourceLayout;

		subResourceLayout = device.getImageSubresourceLayout(dstImage, subResource);

		// Map image memory so we can start copying from it
		const char* data;
		vkMapMemory(device, dstImageMemory, 0, VK_WHOLE_SIZE, 0, (void**)&data);
		data += subResourceLayout.offset;

		std::ofstream file(filename, std::ios::out | std::ios::binary);

		// ppm header
		file << "P6\n" << width << "\n" << height << "\n" << 255 << "\n";

		// If source is BGR (destination is always RGB) and we can't use blit (which does automatic conversion), we'll have to manually swizzle color components
		bool colorSwizzle = false;
		// Check if source is BGR 
		// Note: Not complete, only contains most common and basic BGR surface formats for demonstation purposes
		if (!supportsBlit)
		{
			std::vector<vk::Format> formatsBGR = { vk::Format::eB8G8R8A8Srgb, vk::Format::eB8G8R8A8Unorm, vk::Format::eB8G8R8A8Snorm };
			colorSwizzle = (std::find(formatsBGR.begin(), formatsBGR.end(), swapChain.colorFormat) != formatsBGR.end());
		}

		// ppm binary pixel data
		for (uint32_t y = 0; y < height; y++) 
		{
			unsigned int *row = (unsigned int*)data;
			for (uint32_t x = 0; x < width; x++) 
			{
				if (colorSwizzle) 
				{ 
					file.write((char*)row+2, 1);
					file.write((char*)row+1, 1);
					file.write((char*)row, 1);
				}
				else
				{
					file.write((char*)row, 3);
				}
				row++;
			}
			data += subResourceLayout.rowPitch;
		}
		file.close();

		std::cout << "Screenshot saved to disk" << std::endl;

		// Clean up resources
		device.unmapMemory(dstImageMemory);
		device.freeMemory(dstImageMemory);
		device.destroyImage(dstImage);
	}

	void draw()
	{
		VulkanExampleBase::prepareFrame();

		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &drawCmdBuffers[currentBuffer];
		queue.submit(submitInfo, vk::Fence(nullptr));

		VulkanExampleBase::submitFrame();
	}

	void prepare()
	{
		VulkanExampleBase::prepare();
		loadAssets();
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

	virtual void keyPressed(uint32_t keyCode)
	{
		switch (keyCode)
		{
		case KEY_F2:
		case GAMEPAD_BUTTON_A:
			saveScreenshot("screenshot.ppm");
			break;
		}
	}

	virtual void getOverlayText(VulkanTextOverlay *textOverlay)
	{
#if defined(__ANDROID__)
		textOverlay->addText("\"Button A\" to save screenshot", 5.0f, 85.0f, VulkanTextOverlay::alignLeft);
#else
		textOverlay->addText("\"F2\" to save screenshot", 5.0f, 85.0f, VulkanTextOverlay::alignLeft);
#endif
	}
};

VULKAN_EXAMPLE_MAIN()