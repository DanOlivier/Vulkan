/*
* Vulkan Example - Runtime mip map generation
*
* Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

// todo: Fallback for sampler selection on devices that don't support shaderSampledImageArrayDynamicIndexing

#include "VulkanExampleBase.hpp"
#include "VulkanDevice.hpp"
#include "VulkanModel.hpp"

#include <gli/gli.hpp>

#include <iomanip>
#include <sstream>

#define VERTEX_BUFFER_BIND_ID 0
#define ENABLE_VALIDATION false

class VulkanExample : public VulkanExampleBase
{
public:
	struct Texture {
		vk::Image image;
		vk::ImageLayout imageLayout;
		vk::DeviceMemory deviceMemory;
		vk::ImageView view;
		uint32_t width, height;
		uint32_t mipLevels;
	} texture;

	// To demonstrate mip mapping and filtering this example uses separate samplers
	std::vector<std::string> samplerNames{ "No mip maps" , "With mip maps (bilinear)" , "With mip maps (anisotropic)" };
	std::vector<vk::Sampler> samplers;

	// Vertex layout for the models
	vks::VertexLayout vertexLayout = vks::VertexLayout({
		vks::VERTEX_COMPONENT_POSITION,
		vks::VERTEX_COMPONENT_UV,
		vks::VERTEX_COMPONENT_NORMAL,
	});

	struct {
		vks::Model tunnel;
	} models;

	struct {
		vk::PipelineVertexInputStateCreateInfo inputState;
		std::vector<vk::VertexInputBindingDescription> bindingDescriptions;
		std::vector<vk::VertexInputAttributeDescription> attributeDescriptions;
	} vertices;

	vks::Buffer uniformBufferVS;

	struct uboVS {
		glm::mat4 projection;
		glm::mat4 view;
		glm::mat4 model;
		glm::vec4 viewPos;
		float lodBias = 0.0f;
		uint32_t samplerIndex = 2;
	} uboVS;

	struct {
		vk::Pipeline solid;
	} pipelines;

	vk::PipelineLayout pipelineLayout;
	vk::DescriptorSet descriptorSet;
	vk::DescriptorSetLayout descriptorSetLayout;

	VulkanExample() : VulkanExampleBase(ENABLE_VALIDATION)
	{
		title = "Vulkan Example - Runtime mip map generation";
		enableTextOverlay = true;
		camera.type = Camera::CameraType::firstperson;
		camera.setPerspective(60.0f, (float)width / (float)height, 0.1f, 1024.0f);
		camera.setRotation(glm::vec3(0.0f, 90.0f, 0.0f));
		camera.setTranslation(glm::vec3(40.75f, 0.0f, 0.0f));
		camera.movementSpeed = 2.5f;
		camera.rotationSpeed = 0.5f;
		timerSpeed *= 0.05f;
		paused = true;
	}

	~VulkanExample()
	{
		destroyTextureImage(texture);
		device.destroyPipeline(pipelines.solid);
		device.destroyPipelineLayout(pipelineLayout);
		device.destroyDescriptorSetLayout(descriptorSetLayout);
		uniformBufferVS.destroy();
		for (auto sampler : samplers)
		{
			device.destroySampler(sampler);
		}
		models.tunnel.destroy();
	}

	void loadTexture(std::string fileName, vk::Format format, bool forceLinearTiling)
	{
#if defined(__ANDROID__)
		// Textures are stored inside the apk on Android (compressed)
		// So they need to be loaded via the asset manager
		AAsset* asset = AAssetManager_open(androidApp->activity->assetManager, fileName.c_str(), AASSET_MODE_STREAMING);
		assert(asset);
		size_t size = AAsset_getLength(asset);
		assert(size > 0);

		void *textureData = malloc(size);
		AAsset_read(asset, textureData, size);
		AAsset_close(asset);

		gli::texture2d tex2D(gli::load((const char*)textureData, size));
#else
		gli::texture2d tex2D(gli::load(fileName));
#endif

		assert(!tex2D.empty());

		vk::FormatProperties formatProperties;

		texture.width = static_cast<uint32_t>(tex2D[0].extent().x);
		texture.height = static_cast<uint32_t>(tex2D[0].extent().y);

		// calculate num of mip maps
		// numLevels = 1 + floor(log2(max(w, h, d)))
		// Calculated as log2(max(width, height, depth))c + 1 (see specs)
		texture.mipLevels = floor(log2(std::max(texture.width, texture.height))) + 1;

		// Get device properites for the requested texture format
		formatProperties = physicalDevice.getFormatProperties(format);

		// Mip-chain generation requires support for blit source and destination
		assert(formatProperties.optimalTilingFeatures & vk::FormatFeatureFlagBits::eBlitSrc);
		assert(formatProperties.optimalTilingFeatures & vk::FormatFeatureFlagBits::eBlitDst);

		vk::MemoryAllocateInfo memAllocInfo = vks::initializers::memoryAllocateInfo();
		vk::MemoryRequirements memReqs = {};

		// Create a host-visible staging buffer that contains the raw image data
		vk::Buffer stagingBuffer;
		vk::DeviceMemory stagingMemory;

		vk::BufferCreateInfo bufferCreateInfo = vks::initializers::bufferCreateInfo();
		bufferCreateInfo.size = tex2D.size();
		// This buffer is used as a transfer source for the buffer copy
		bufferCreateInfo.usage = vk::BufferUsageFlagBits::eTransferSrc;
		bufferCreateInfo.sharingMode = vk::SharingMode::eExclusive;		
		stagingBuffer = device.createBuffer(bufferCreateInfo);
		memReqs = device.getBufferMemoryRequirements(stagingBuffer);
		memAllocInfo.allocationSize = memReqs.size;
		memAllocInfo.memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
		stagingMemory = device.allocateMemory(memAllocInfo);
		device.bindBufferMemory(stagingBuffer, stagingMemory, 0);

		// Copy texture data into staging buffer
		uint8_t *data = (uint8_t*)device.mapMemory(stagingMemory, 0, memReqs.size, vk::MemoryMapFlags());
		memcpy(data, tex2D.data(), tex2D.size());
		device.unmapMemory(stagingMemory);

		// Create optimal tiled target image
		vk::ImageCreateInfo imageCreateInfo = vks::initializers::imageCreateInfo();
		imageCreateInfo.imageType = vk::ImageType::e2D;
		imageCreateInfo.format = format;
		imageCreateInfo.mipLevels = texture.mipLevels;
		imageCreateInfo.arrayLayers = 1;
		imageCreateInfo.samples = vk::SampleCountFlagBits::e1;
		imageCreateInfo.tiling = vk::ImageTiling::eOptimal;
		imageCreateInfo.usage = vk::ImageUsageFlagBits::eSampled;
		imageCreateInfo.sharingMode = vk::SharingMode::eExclusive;
		imageCreateInfo.initialLayout = vk::ImageLayout::eUndefined;
		imageCreateInfo.extent = vk::Extent3D{ texture.width, texture.height, 1 };
		imageCreateInfo.usage = vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eSampled;
		texture.image = device.createImage(imageCreateInfo);
		memReqs = device.getImageMemoryRequirements(texture.image);
		memAllocInfo.allocationSize = memReqs.size;
		memAllocInfo.memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);
		texture.deviceMemory = device.allocateMemory(memAllocInfo);
		device.bindImageMemory(texture.image, texture.deviceMemory, 0);

		vk::CommandBuffer copyCmd = VulkanExampleBase::createCommandBuffer(vk::CommandBufferLevel::ePrimary, true);

		vk::ImageSubresourceRange subresourceRange;
		subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
		subresourceRange.levelCount = 1;
		subresourceRange.layerCount = 1;

		// Optimal image will be used as destination for the copy, so we must transfer from our initial undefined image layout to the transfer destination layout
		vks::tools::setImageLayout(
			copyCmd,
			texture.image,
			vk::ImageLayout::eUndefined,
			vk::ImageLayout::eTransferDstOptimal,
			subresourceRange);

		// Copy the first mip of the chain, remaining mips will be generated
		vk::BufferImageCopy bufferCopyRegion = {};
		bufferCopyRegion.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
		bufferCopyRegion.imageSubresource.mipLevel = 0;
		bufferCopyRegion.imageSubresource.baseArrayLayer = 0;
		bufferCopyRegion.imageSubresource.layerCount = 1;
		bufferCopyRegion.imageExtent.width = texture.width;
		bufferCopyRegion.imageExtent.height = texture.height;
		bufferCopyRegion.imageExtent.depth = 1;

		copyCmd.copyBufferToImage(stagingBuffer, texture.image, vk::ImageLayout::eTransferDstOptimal, bufferCopyRegion);

		// Transition first mip level to transfer source for read during blit
		texture.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
		vks::tools::setImageLayout(
			copyCmd,
			texture.image,
			vk::ImageLayout::eTransferDstOptimal,
			vk::ImageLayout::eTransferSrcOptimal,
			subresourceRange);

		VulkanExampleBase::flushCommandBuffer(copyCmd, queue, true);

		// Clean up staging resources
		device.freeMemory(stagingMemory);
		device.destroyBuffer(stagingBuffer);

		// Generate the mip chain
		// ---------------------------------------------------------------
		// We copy down the whole mip chain doing a blit from mip-1 to mip
		// An alternative way would be to always blit from the first mip level and sample that one down
		vk::CommandBuffer blitCmd = VulkanExampleBase::createCommandBuffer(vk::CommandBufferLevel::ePrimary, true);

		// Copy down mips from n-1 to n
		for (uint32_t i = 1; i < texture.mipLevels; i++)
		{
			vk::ImageBlit imageBlit{};				

			// Source
			imageBlit.srcSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
			imageBlit.srcSubresource.layerCount = 1;
			imageBlit.srcSubresource.mipLevel = i-1;
			imageBlit.srcOffsets[1].x = int32_t(texture.width >> (i - 1));
			imageBlit.srcOffsets[1].y = int32_t(texture.height >> (i - 1));
			imageBlit.srcOffsets[1].z = 1;

			// Destination
			imageBlit.dstSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
			imageBlit.dstSubresource.layerCount = 1;
			imageBlit.dstSubresource.mipLevel = i;
			imageBlit.dstOffsets[1].x = int32_t(texture.width >> i);
			imageBlit.dstOffsets[1].y = int32_t(texture.height >> i);
			imageBlit.dstOffsets[1].z = 1;

			vk::ImageSubresourceRange mipSubRange = {};
			mipSubRange.aspectMask = vk::ImageAspectFlagBits::eColor;
			mipSubRange.baseMipLevel = i;
			mipSubRange.levelCount = 1;
			mipSubRange.layerCount = 1;

			// Transiton current mip level to transfer dest
			vks::tools::setImageLayout(
				blitCmd,
				texture.image,
				vk::ImageLayout::eUndefined,
				vk::ImageLayout::eTransferDstOptimal,
				mipSubRange,
				vk::PipelineStageFlagBits::eTransfer,
				vk::PipelineStageFlagBits::eHost);

			// Blit from previous level
			blitCmd.blitImage(
				texture.image,
				vk::ImageLayout::eTransferSrcOptimal,
				texture.image,
				vk::ImageLayout::eTransferDstOptimal,
				imageBlit,
				vk::Filter::eLinear);

			// Transiton current mip level to transfer source for read in next iteration
			vks::tools::setImageLayout(
				blitCmd,
				texture.image,
				vk::ImageLayout::eTransferDstOptimal,
				vk::ImageLayout::eTransferSrcOptimal,
				mipSubRange,
				vk::PipelineStageFlagBits::eHost,
				vk::PipelineStageFlagBits::eTransfer);
		}

		// After the loop, all mip layers are in TRANSFER_SRC layout, so transition all to SHADER_READ
		subresourceRange.levelCount = texture.mipLevels;
		vks::tools::setImageLayout(
			blitCmd,
			texture.image,
			vk::ImageLayout::eTransferSrcOptimal,
			texture.imageLayout,
			subresourceRange);

		VulkanExampleBase::flushCommandBuffer(blitCmd, queue, true);
		// ---------------------------------------------------------------

		// Create samplers
		samplers.resize(3);
		vk::SamplerCreateInfo sampler = vks::initializers::samplerCreateInfo();
		sampler.magFilter = vk::Filter::eLinear;
		sampler.minFilter = vk::Filter::eLinear;
		sampler.mipmapMode = vk::SamplerMipmapMode::eLinear;
		sampler.addressModeU = vk::SamplerAddressMode::eMirroredRepeat;
		sampler.addressModeV = vk::SamplerAddressMode::eMirroredRepeat;
		sampler.addressModeW = vk::SamplerAddressMode::eMirroredRepeat;
		sampler.mipLodBias = 0.0f;
		sampler.compareOp = vk::CompareOp::eNever;
		sampler.minLod = 0.0f;
		sampler.maxLod = 0.0f;
		sampler.borderColor = vk::BorderColor::eFloatOpaqueWhite;
		sampler.maxAnisotropy = 1.0;
		sampler.anisotropyEnable = VK_FALSE;

		// Without mip mapping
		samplers[0] = device.createSampler(sampler);

		// With mip mapping
		sampler.maxLod = (float)texture.mipLevels;
		samplers[1] = device.createSampler(sampler);

		// With mip mapping and anisotropic filtering
		if (vulkanDevice->features.samplerAnisotropy)
		{
			sampler.maxAnisotropy = vulkanDevice->properties.limits.maxSamplerAnisotropy;
			sampler.anisotropyEnable = VK_TRUE;
		}
		samplers[2] = device.createSampler(sampler);

		// Create image view
		vk::ImageViewCreateInfo view = vks::initializers::imageViewCreateInfo();
		view.image = texture.image;
		view.viewType = vk::ImageViewType::e2D;
		view.format = format;
		view.components = { vk::ComponentSwizzle::eR, vk::ComponentSwizzle::eG, vk::ComponentSwizzle::eB, vk::ComponentSwizzle::eA };
		view.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
		view.subresourceRange.baseMipLevel = 0;
		view.subresourceRange.baseArrayLayer = 0;
		view.subresourceRange.layerCount = 1;
		view.subresourceRange.levelCount = texture.mipLevels;
		texture.view = device.createImageView(view);
	}

	// Free all Vulkan resources used a texture object
	void destroyTextureImage(Texture texture)
	{
		device.destroyImageView(texture.view);
		device.destroyImage(texture.image);
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
			drawCmdBuffers[i].bindVertexBuffers(VERTEX_BUFFER_BIND_ID, models.tunnel.vertices.buffer, offsets);
			drawCmdBuffers[i].bindIndexBuffer(models.tunnel.indices.buffer, 0, vk::IndexType::eUint32);

			drawCmdBuffers[i].drawIndexed(models.tunnel.indexCount, 1, 0, 0, 0);

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

	void loadAssets()
	{
		models.tunnel.loadFromFile(getAssetPath() + "models/tunnel_cylinder.dae", vertexLayout, 1.0f, vulkanDevice, queue);
		loadTexture(getAssetPath() + "textures/metalplate_nomips_rgba.ktx", vk::Format::eR8G8B8A8Unorm, false);
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
		// Describes memory layout and shader positions
		vertices.attributeDescriptions.resize(3);
		// Location 0 : Position
		vertices.attributeDescriptions[0] =
			vks::initializers::vertexInputAttributeDescription(
				VERTEX_BUFFER_BIND_ID,
				0,
				vk::Format::eR32G32B32Sfloat,
				0);			
		// Location 1 : Texture coordinates
		vertices.attributeDescriptions[1] =
			vks::initializers::vertexInputAttributeDescription(
				VERTEX_BUFFER_BIND_ID,
				1,
				vk::Format::eR32G32Sfloat,
				3 * sizeof(float));
		// Location 1 : Vertex normal
		vertices.attributeDescriptions[2] =
			vks::initializers::vertexInputAttributeDescription(
				VERTEX_BUFFER_BIND_ID,
				2,
				vk::Format::eR32G32B32Sfloat,
				5 * sizeof(float));

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
			vks::initializers::descriptorPoolSize(vk::DescriptorType::eUniformBuffer, 1),	// Vertex shader UBO
			vks::initializers::descriptorPoolSize(vk::DescriptorType::eSampledImage, 1),		// Sampled image
			vks::initializers::descriptorPoolSize(vk::DescriptorType::eSampler, 3),			// 3 samplers (array)
		};

		vk::DescriptorPoolCreateInfo descriptorPoolInfo = 
			vks::initializers::descriptorPoolCreateInfo(
				static_cast<uint32_t>(poolSizes.size()),
				poolSizes.data(),
				1);

		descriptorPool = device.createDescriptorPool(descriptorPoolInfo);
	}

	void setupDescriptorSetLayout()
	{
		std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings;

		// Binding 0: Vertex shader uniform buffer
		setLayoutBindings.push_back(vks::initializers::descriptorSetLayoutBinding(
			vk::DescriptorType::eUniformBuffer,
			vk::ShaderStageFlagBits::eVertex,
			0));

		// Binding 1: Sampled image
		setLayoutBindings.push_back(vks::initializers::descriptorSetLayoutBinding(
			vk::DescriptorType::eSampledImage,
			vk::ShaderStageFlagBits::eFragment,
			1));

		// Binding 2: Sampler array (3 descriptors)
		setLayoutBindings.push_back(vks::initializers::descriptorSetLayoutBinding(
			vk::DescriptorType::eSampler,
			vk::ShaderStageFlagBits::eFragment,
			2,
			3));

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

		std::vector<vk::WriteDescriptorSet> writeDescriptorSets;

		// Binding 0: Vertex shader uniform buffer
		writeDescriptorSets.push_back(vks::initializers::writeDescriptorSet(
			descriptorSet,
			vk::DescriptorType::eUniformBuffer,
			0,
			&uniformBufferVS.descriptor));

		// Binding 1: Sampled image
		vk::DescriptorImageInfo textureDescriptor = 
			vks::initializers::descriptorImageInfo(
				nullptr,				 
				texture.view, 
				texture.imageLayout);
		writeDescriptorSets.push_back(vks::initializers::writeDescriptorSet(
			descriptorSet,
			vk::DescriptorType::eSampledImage,
			1,
			&textureDescriptor));

		// Binding 2: Sampler array
		std::vector<vk::DescriptorImageInfo> samplerDescriptors;
		for (uint32_t i = 0; i < samplers.size(); i++)
		{
			samplerDescriptors.push_back(vks::initializers::descriptorImageInfo(samplers[i], nullptr, vk::ImageLayout::eShaderReadOnlyOptimal));
		}
		vk::WriteDescriptorSet samplerDescriptorWrite{};

		samplerDescriptorWrite.dstSet = descriptorSet;
		samplerDescriptorWrite.descriptorType = vk::DescriptorType::eSampler;
		samplerDescriptorWrite.descriptorCount = static_cast<uint32_t>(samplerDescriptors.size());
		samplerDescriptorWrite.pImageInfo = samplerDescriptors.data();
		samplerDescriptorWrite.dstBinding = 2;
		samplerDescriptorWrite.dstArrayElement = 0;
		writeDescriptorSets.push_back(samplerDescriptorWrite);

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

		shaderStages[0] = loadShader(getAssetPath() + "shaders/texturemipmapgen/texture.vert.spv", vk::ShaderStageFlagBits::eVertex);
		shaderStages[1] = loadShader(getAssetPath() + "shaders/texturemipmapgen/texture.frag.spv", vk::ShaderStageFlagBits::eFragment);

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
		uboVS.projection = camera.matrices.perspective;
		uboVS.view = camera.matrices.view;
		uboVS.model = glm::rotate(glm::mat4(), glm::radians(timer * 360.0f), glm::vec3(1.0f, 0.0f, 0.0f));
		uboVS.viewPos = glm::vec4(camera.position, 0.0f) * glm::vec4(-1.0f);
		uniformBufferVS.map();
		memcpy(uniformBufferVS.mapped, &uboVS, sizeof(uboVS));
		uniformBufferVS.unmap();
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
		prepared = true;
	}

	virtual void render()
	{
		if (!prepared)
			return;
		draw();
		if (!paused)
		{
			updateUniformBuffers();
		}
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

	void toggleSampler()
	{
		uboVS.samplerIndex = (uboVS.samplerIndex < static_cast<uint32_t>(samplers.size()) - 1) ? uboVS.samplerIndex + 1 : 0;
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
		case KEY_F:
		case GAMEPAD_BUTTON_A:
			toggleSampler();
			break;
		}
	}

	virtual void getOverlayText(VulkanTextOverlay *textOverlay)
	{
		std::stringstream ss;
		ss << std::setprecision(2) << std::fixed << uboVS.lodBias;
#if defined(__ANDROID__)
		textOverlay->addText("LOD bias: " + ss.str() + " (Buttons L1/R1 to change)", 5.0f, 85.0f, VulkanTextOverlay::alignLeft);
		textOverlay->addText("Sampler: " + samplerNames[uboVS.samplerIndex] + " (\"Button A\" to toggle)", 5.0f, 105.0f, VulkanTextOverlay::alignLeft);
#else
		textOverlay->addText("LOD bias: " + ss.str() + " (numpad +/- to change)", 5.0f, 85.0f, VulkanTextOverlay::alignLeft);
		textOverlay->addText("Sampler: " + samplerNames[uboVS.samplerIndex] + " (\"f\" to toggle)", 5.0f, 105.0f, VulkanTextOverlay::alignLeft);
#endif
	}
};

VULKAN_EXAMPLE_MAIN()
