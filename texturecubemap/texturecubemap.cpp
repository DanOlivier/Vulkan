/*
* Vulkan Example - Cube map texture loading and displaying
*
* Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include "vulkanexamplebase.h"
#include "VulkanTexture.hpp"
#include "VulkanModel.hpp"

#include <iomanip>
#include <sstream>

#define VERTEX_BUFFER_BIND_ID 0
#define ENABLE_VALIDATION false

class VulkanExample : public VulkanExampleBase
{
public:
	bool displaySkybox = true;

	vks::Texture cubeMap;

	struct {
		vk::PipelineVertexInputStateCreateInfo inputState;
		std::vector<vk::VertexInputBindingDescription> bindingDescriptions;
		std::vector<vk::VertexInputAttributeDescription> attributeDescriptions;
	} vertices;

	// Vertex layout for the models
	vks::VertexLayout vertexLayout = vks::VertexLayout({
		vks::VERTEX_COMPONENT_POSITION,
		vks::VERTEX_COMPONENT_NORMAL,
		vks::VERTEX_COMPONENT_UV,
	});

	struct Meshes {
		vks::Model skybox;
		std::vector<vks::Model> objects;
		uint32_t objectIndex = 0;
	} models;

	struct {
		vks::Buffer object;
		vks::Buffer skybox;
	} uniformBuffers;

	struct UBOVS {
		glm::mat4 projection;
		glm::mat4 model;
		float lodBias = 0.0f;
	} uboVS;

	struct {
		vk::Pipeline skybox;
		vk::Pipeline reflect;
	} pipelines;

	struct {
		vk::DescriptorSet object;
		vk::DescriptorSet skybox;
	} descriptorSets;

	vk::PipelineLayout pipelineLayout;
	vk::DescriptorSetLayout descriptorSetLayout;

	VulkanExample() : VulkanExampleBase(ENABLE_VALIDATION)
	{
		zoom = -4.0f;
		rotationSpeed = 0.25f;
		rotation = { -7.25f, -120.0f, 0.0f };
		enableTextOverlay = true;
		title = "Vulkan Example - Cube map";
	}

	~VulkanExample()
	{
		// Clean up used Vulkan resources 
		// Note : Inherited destructor cleans up resources stored in base class

		// Clean up texture resources
		device.destroyImageView(cubeMap.view);
		device.destroyImage(cubeMap.image);
		device.destroySampler(cubeMap.sampler);
		device.freeMemory(cubeMap.deviceMemory);

		device.destroyPipeline(pipelines.skybox);
		device.destroyPipeline(pipelines.reflect);

		device.destroyPipelineLayout(pipelineLayout);
		device.destroyDescriptorSetLayout(descriptorSetLayout);

		for (auto& model : models.objects) {
			model.destroy();
		}
		models.skybox.destroy();

		uniformBuffers.object.destroy();
		uniformBuffers.skybox.destroy();
	}

	void loadCubemap(std::string filename, vk::Format format, bool forceLinearTiling)
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
#else
		gli::texture_cube texCube(gli::load(filename));
#endif

		assert(!texCube.empty());

		cubeMap.width = texCube.extent().x;
		cubeMap.height = texCube.extent().y;
		cubeMap.mipLevels = texCube.levels();

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
		memcpy(data, texCube.data(), texCube.size());
		device.unmapMemory(stagingMemory);

		// Create optimal tiled target image
		vk::ImageCreateInfo imageCreateInfo = vks::initializers::imageCreateInfo();
		imageCreateInfo.imageType = vk::ImageType::e2D;
		imageCreateInfo.format = format;
		imageCreateInfo.mipLevels = cubeMap.mipLevels;
		imageCreateInfo.samples = vk::SampleCountFlagBits::e1;
		imageCreateInfo.tiling = vk::ImageTiling::eOptimal;
		imageCreateInfo.usage = vk::ImageUsageFlagBits::eSampled;
		imageCreateInfo.sharingMode = vk::SharingMode::eExclusive;
		imageCreateInfo.initialLayout = vk::ImageLayout::eUndefined;
		imageCreateInfo.extent = vk::Extent3D{ cubeMap.width, cubeMap.height, 1 };
		imageCreateInfo.usage = vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled;
		// Cube faces count as array layers in Vulkan
		imageCreateInfo.arrayLayers = 6;
		// This flag is required for cube map images
		imageCreateInfo.flags = vk::ImageCreateFlagBits::eCubeCompatible;

		cubeMap.image = device.createImage(imageCreateInfo);

		memReqs = device.getImageMemoryRequirements(cubeMap.image);

		memAllocInfo.allocationSize = memReqs.size;
		memAllocInfo.memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);

		cubeMap.deviceMemory = device.allocateMemory(memAllocInfo);
		device.bindImageMemory(cubeMap.image, cubeMap.deviceMemory, 0);

		vk::CommandBuffer copyCmd = VulkanExampleBase::createCommandBuffer(vk::CommandBufferLevel::ePrimary, true);

		// Setup buffer copy regions for each face including all of it's miplevels
		std::vector<vk::BufferImageCopy> bufferCopyRegions;
		uint32_t offset = 0;

		for (uint32_t face = 0; face < 6; face++)
		{
			for (uint32_t level = 0; level < cubeMap.mipLevels; level++)
			{
				vk::BufferImageCopy bufferCopyRegion = {};
				bufferCopyRegion.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
				bufferCopyRegion.imageSubresource.mipLevel = level;
				bufferCopyRegion.imageSubresource.baseArrayLayer = face;
				bufferCopyRegion.imageSubresource.layerCount = 1;
				bufferCopyRegion.imageExtent.width = texCube[face][level].extent().x;
				bufferCopyRegion.imageExtent.height = texCube[face][level].extent().y;
				bufferCopyRegion.imageExtent.depth = 1;
				bufferCopyRegion.bufferOffset = offset;

				bufferCopyRegions.push_back(bufferCopyRegion);

				// Increase offset into staging buffer for next level / face
				offset += texCube[face][level].size();
			}
		}

		// Image barrier for optimal image (target)
		// Set initial layout for all array layers (faces) of the optimal (target) tiled texture
		vk::ImageSubresourceRange subresourceRange;
		subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
		subresourceRange.baseMipLevel = 0;
		subresourceRange.levelCount = cubeMap.mipLevels;
		subresourceRange.layerCount = 6;

		vks::tools::setImageLayout(
			copyCmd,
			cubeMap.image,
			vk::ImageLayout::eUndefined,
			vk::ImageLayout::eTransferDstOptimal,
			subresourceRange);

		// Copy the cube map faces from the staging buffer to the optimal tiled image
		copyCmd.copyBufferToImage(
			stagingBuffer,
			cubeMap.image,
			vk::ImageLayout::eTransferDstOptimal,
			bufferCopyRegions);

		// Change texture image layout to shader read after all faces have been copied
		cubeMap.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
		vks::tools::setImageLayout(
			copyCmd,
			cubeMap.image,
			vk::ImageLayout::eTransferDstOptimal,
			cubeMap.imageLayout,
			subresourceRange);

		VulkanExampleBase::flushCommandBuffer(copyCmd, queue, true);

		// Create sampler
		vk::SamplerCreateInfo sampler = vks::initializers::samplerCreateInfo();
		sampler.magFilter = vk::Filter::eLinear;
		sampler.minFilter = vk::Filter::eLinear;
		sampler.mipmapMode = vk::SamplerMipmapMode::eLinear;
		sampler.addressModeU = vk::SamplerAddressMode::eClampToEdge;
		sampler.addressModeV = sampler.addressModeU;
		sampler.addressModeW = sampler.addressModeU;
		sampler.mipLodBias = 0.0f;
		sampler.compareOp = vk::CompareOp::eNever;
		sampler.minLod = 0.0f;
		sampler.maxLod = cubeMap.mipLevels;
		sampler.borderColor = vk::BorderColor::eFloatOpaqueWhite;
		sampler.maxAnisotropy = 1.0f;
		if (vulkanDevice->features.samplerAnisotropy)
		{
			sampler.maxAnisotropy = vulkanDevice->properties.limits.maxSamplerAnisotropy;
			sampler.anisotropyEnable = VK_TRUE;
		}
		cubeMap.sampler = device.createSampler(sampler);

		// Create image view
		vk::ImageViewCreateInfo view = vks::initializers::imageViewCreateInfo();
		// Cube map view type
		view.viewType = vk::ImageViewType::eCube;
		view.format = format;
		view.components = { vk::ComponentSwizzle::eR, vk::ComponentSwizzle::eG, vk::ComponentSwizzle::eB, vk::ComponentSwizzle::eA };
		view.subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 };
		// 6 array layers (faces)
		view.subresourceRange.layerCount = 6;
		// Set number of mip levels
		view.subresourceRange.levelCount = cubeMap.mipLevels;
		view.image = cubeMap.image;
		cubeMap.view = device.createImageView(view);

		// Clean up staging resources
		device.freeMemory(stagingMemory);
		device.destroyBuffer(stagingBuffer);
	}

	void loadTextures()
	{
		// Vulkan core supports three different compressed texture formats
		// As the support differs between implemementations we need to check device features and select a proper format and file
		std::string filename;
		vk::Format format;
		if (deviceFeatures.textureCompressionBC) {
			filename = "cubemap_yokohama_bc3_unorm.ktx";
			format = vk::Format::eBc2UnormBlock;
		}
		else if (deviceFeatures.textureCompressionASTC_LDR) {
			filename = "cubemap_yokohama_astc_8x8_unorm.ktx";
			format = vk::Format::eAstc8x8UnormBlock;
		}
		else if (deviceFeatures.textureCompressionETC2) {
			filename = "cubemap_yokohama_etc2_unorm.ktx";
			format = vk::Format::eEtc2R8G8B8UnormBlock;
		}
		else {
			vks::tools::exitFatal("Device does not support any compressed texture format!", "Error");
		}

		loadCubemap(getAssetPath() + "textures/" + filename, format, false);
	}

	void reBuildCommandBuffers()
	{
		if (!checkCommandBuffers())
		{
			destroyCommandBuffers();
			createCommandBuffers();
		}
		buildCommandBuffers();
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

			vk::Viewport viewport = vks::initializers::viewport((float)width,	(float)height, 0.0f, 1.0f);
			drawCmdBuffers[i].setViewport(0, viewport);

			vk::Rect2D scissor = vks::initializers::rect2D(width,	height,	0, 0);
			drawCmdBuffers[i].setScissor(0, scissor);

			std::vector<vk::DeviceSize> offsets = { 0 };

			// Skybox
			if (displaySkybox)
			{
				drawCmdBuffers[i].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSets.skybox, nullptr);
				drawCmdBuffers[i].bindVertexBuffers(VERTEX_BUFFER_BIND_ID, models.skybox.vertices.buffer, offsets);
				drawCmdBuffers[i].bindIndexBuffer(models.skybox.indices.buffer, 0, vk::IndexType::eUint32);
				drawCmdBuffers[i].bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.skybox);
				drawCmdBuffers[i].drawIndexed(models.skybox.indexCount, 1, 0, 0, 0);
			}

			// 3D object
			drawCmdBuffers[i].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSets.object, nullptr);
			drawCmdBuffers[i].bindVertexBuffers(VERTEX_BUFFER_BIND_ID, models.objects[models.objectIndex].vertices.buffer, offsets);
			drawCmdBuffers[i].bindIndexBuffer(models.objects[models.objectIndex].indices.buffer, 0, vk::IndexType::eUint32);
			drawCmdBuffers[i].bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.reflect);
			drawCmdBuffers[i].drawIndexed(models.objects[models.objectIndex].indexCount, 1, 0, 0, 0);

			drawCmdBuffers[i].endRenderPass();

			drawCmdBuffers[i].end();
		}
	}

	void loadMeshes()
	{
		// Skybox
		models.skybox.loadFromFile(getAssetPath() + "models/cube.obj", vertexLayout, 0.05f, vulkanDevice, queue);
		// Objects
		std::vector<std::string> filenames = { "sphere.obj", "teapot.dae", "torusknot.obj" };
		for (auto file : filenames) {
			vks::Model model;
			model.loadFromFile(getAssetPath() + "models/" + file, vertexLayout, 0.05f, vulkanDevice, queue);
			models.objects.push_back(model);
		}
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
		// Location 1 : Normal
		vertices.attributeDescriptions[1] =
			vks::initializers::vertexInputAttributeDescription(
				VERTEX_BUFFER_BIND_ID,
				1,
				vk::Format::eR32G32B32Sfloat,
				sizeof(float) * 3);
		// Location 2 : Texture coordinates
		vertices.attributeDescriptions[2] =
			vks::initializers::vertexInputAttributeDescription(
				VERTEX_BUFFER_BIND_ID,
				2,
				vk::Format::eR32G32Sfloat,
				sizeof(float) * 5);

		vertices.inputState = vks::initializers::pipelineVertexInputStateCreateInfo();
		vertices.inputState.vertexBindingDescriptionCount = vertices.bindingDescriptions.size();
		vertices.inputState.pVertexBindingDescriptions = vertices.bindingDescriptions.data();
		vertices.inputState.vertexAttributeDescriptionCount = vertices.attributeDescriptions.size();
		vertices.inputState.pVertexAttributeDescriptions = vertices.attributeDescriptions.data();
	}

	void setupDescriptorPool()
	{
		std::vector<vk::DescriptorPoolSize> poolSizes =
		{
			vks::initializers::descriptorPoolSize(vk::DescriptorType::eUniformBuffer, 2),
			vks::initializers::descriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, 2)
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
				setLayoutBindings.data(),
				setLayoutBindings.size());

		descriptorSetLayout = device.createDescriptorSetLayout(descriptorLayout);

		vk::PipelineLayoutCreateInfo pPipelineLayoutCreateInfo =
			vks::initializers::pipelineLayoutCreateInfo(
				&descriptorSetLayout,
				1);

		pipelineLayout = device.createPipelineLayout(pPipelineLayoutCreateInfo);
	}

	void setupDescriptorSets()
	{
		// Image descriptor for the cube map texture
		vk::DescriptorImageInfo textureDescriptor =
			vks::initializers::descriptorImageInfo(
				cubeMap.sampler,
				cubeMap.view,
				cubeMap.imageLayout);

		vk::DescriptorSetAllocateInfo allocInfo =
			vks::initializers::descriptorSetAllocateInfo(
				descriptorPool,
				&descriptorSetLayout,
				1);

		// 3D object descriptor set
		descriptorSets.object = device.allocateDescriptorSets(allocInfo)[0];

		std::vector<vk::WriteDescriptorSet> writeDescriptorSets =
		{
			// Binding 0 : Vertex shader uniform buffer
			vks::initializers::writeDescriptorSet(
				descriptorSets.object, 
				vk::DescriptorType::eUniformBuffer, 
				0, 
				&uniformBuffers.object.descriptor),
			// Binding 1 : Fragment shader cubemap sampler
			vks::initializers::writeDescriptorSet(
				descriptorSets.object, 
				vk::DescriptorType::eCombinedImageSampler, 
				1, 
				&textureDescriptor)
		};
		device.updateDescriptorSets(writeDescriptorSets, nullptr);

		// Sky box descriptor set
		descriptorSets.skybox = device.allocateDescriptorSets(allocInfo)[0];

		writeDescriptorSets =
		{
			// Binding 0 : Vertex shader uniform buffer
			vks::initializers::writeDescriptorSet(
				descriptorSets.skybox,
				vk::DescriptorType::eUniformBuffer,
				0,
				&uniformBuffers.skybox.descriptor),
			// Binding 1 : Fragment shader cubemap sampler
			vks::initializers::writeDescriptorSet(
				descriptorSets.skybox,
				vk::DescriptorType::eCombinedImageSampler,
				1,
				&textureDescriptor)
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
				VK_FALSE,
				VK_FALSE,
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

		// Skybox pipeline (background cube)
		std::array<vk::PipelineShaderStageCreateInfo,2> shaderStages;

		shaderStages[0] = loadShader(getAssetPath() + "shaders/cubemap/skybox.vert.spv", vk::ShaderStageFlagBits::eVertex);
		shaderStages[1] = loadShader(getAssetPath() + "shaders/cubemap/skybox.frag.spv", vk::ShaderStageFlagBits::eFragment);

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
		pipelineCreateInfo.stageCount = shaderStages.size();
		pipelineCreateInfo.pStages = shaderStages.data();

		pipelines.skybox = device.createGraphicsPipelines(pipelineCache, pipelineCreateInfo)[0];

		// Cube map reflect pipeline
		shaderStages[0] = loadShader(getAssetPath() + "shaders/cubemap/reflect.vert.spv", vk::ShaderStageFlagBits::eVertex);
		shaderStages[1] = loadShader(getAssetPath() + "shaders/cubemap/reflect.frag.spv", vk::ShaderStageFlagBits::eFragment);
		// Enable depth test and write
		depthStencilState.depthWriteEnable = VK_TRUE;
		depthStencilState.depthTestEnable = VK_TRUE;
		// Flip cull mode
		rasterizationState.cullMode = vk::CullModeFlagBits::eFront;
		pipelines.reflect = device.createGraphicsPipelines(pipelineCache, pipelineCreateInfo)[0];
	}

	// Prepare and initialize uniform buffer containing shader uniforms
	void prepareUniformBuffers()
	{
		// Objact vertex shader uniform buffer
		vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eUniformBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			&uniformBuffers.object,
			sizeof(uboVS));

		// Skybox vertex shader uniform buffer
		vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eUniformBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			&uniformBuffers.skybox,
			sizeof(uboVS));

		// Map persistent
		uniformBuffers.object.map();
		uniformBuffers.skybox.map();

		updateUniformBuffers();
	}

	void updateUniformBuffers()
	{
		// 3D object
		glm::mat4 viewMatrix = glm::mat4();
		uboVS.projection = glm::perspective(glm::radians(60.0f), (float)width / (float)height, 0.001f, 256.0f);
		viewMatrix = glm::translate(viewMatrix, glm::vec3(0.0f, 0.0f, zoom));

		uboVS.model = glm::mat4();
		uboVS.model = viewMatrix * glm::translate(uboVS.model, cameraPos);
		uboVS.model = glm::rotate(uboVS.model, glm::radians(rotation.x), glm::vec3(1.0f, 0.0f, 0.0f));
		uboVS.model = glm::rotate(uboVS.model, glm::radians(rotation.y), glm::vec3(0.0f, 1.0f, 0.0f));
		uboVS.model = glm::rotate(uboVS.model, glm::radians(rotation.z), glm::vec3(0.0f, 0.0f, 1.0f));

		memcpy(uniformBuffers.object.mapped, &uboVS, sizeof(uboVS));

		// Skybox
		viewMatrix = glm::mat4();
		uboVS.projection = glm::perspective(glm::radians(60.0f), (float)width / (float)height, 0.001f, 256.0f);

		uboVS.model = glm::mat4();
		uboVS.model = viewMatrix * glm::translate(uboVS.model, glm::vec3(0, 0, 0));
		uboVS.model = glm::rotate(uboVS.model, glm::radians(rotation.x), glm::vec3(1.0f, 0.0f, 0.0f));
		uboVS.model = glm::rotate(uboVS.model, glm::radians(rotation.y), glm::vec3(0.0f, 1.0f, 0.0f));
		uboVS.model = glm::rotate(uboVS.model, glm::radians(rotation.z), glm::vec3(0.0f, 0.0f, 1.0f));

		memcpy(uniformBuffers.skybox.mapped, &uboVS, sizeof(uboVS));
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
		loadTextures();
		loadMeshes();
		setupVertexDescriptions();
		prepareUniformBuffers();
		setupDescriptorSetLayout();
		preparePipelines();
		setupDescriptorPool();
		setupDescriptorSets();
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

	void toggleSkyBox()
	{
		displaySkybox = !displaySkybox;
		reBuildCommandBuffers();
	}

	void toggleObject()
	{
		models.objectIndex++;
		if (models.objectIndex >= static_cast<uint32_t>(models.objects.size()))
		{
			models.objectIndex = 0;
		}
		reBuildCommandBuffers();
	}

	void changeLodBias(float delta)
	{
		uboVS.lodBias += delta;
		if (uboVS.lodBias < 0.0f)
		{
			uboVS.lodBias = 0.0f;
		}
		if (uboVS.lodBias > cubeMap.mipLevels)
		{
			uboVS.lodBias = cubeMap.mipLevels;
		}
		updateUniformBuffers();
		updateTextOverlay();
	}

	virtual void keyPressed(uint32_t keyCode)
	{
		switch (keyCode)
		{
		case KEY_S:
		case GAMEPAD_BUTTON_A:
			toggleSkyBox();
			break;
		case KEY_SPACE:
		case GAMEPAD_BUTTON_X:
			toggleObject();
			break;
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
		std::stringstream ss;
		ss << std::setprecision(2) << std::fixed << uboVS.lodBias;
#if defined(__ANDROID__)
		textOverlay->addText("Press \"Button A\" to toggle skybox", 5.0f, 85.0f, VulkanTextOverlay::alignLeft);
		textOverlay->addText("Press \"Button X\" to toggle object", 5.0f, 100.0f, VulkanTextOverlay::alignLeft);
		textOverlay->addText("LOD bias: " + ss.str() + " (Buttons L1/R1 to change)", 5.0f, 115.0f, VulkanTextOverlay::alignLeft);
#else
		textOverlay->addText("Press \"s\" to toggle skybox", 5.0f, 85.0f, VulkanTextOverlay::alignLeft);
		textOverlay->addText("Press \"space\" to toggle object", 5.0f, 100.0f, VulkanTextOverlay::alignLeft);
		textOverlay->addText("LOD bias: " + ss.str() + " (numpad +/- to change)", 5.0f, 115.0f, VulkanTextOverlay::alignLeft);
#endif
	}
};

VULKAN_EXAMPLE_MAIN()