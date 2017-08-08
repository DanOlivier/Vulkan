/*
* Vulkan Example - Physical based rendering a textured object (metal/roughness workflow) with image based lighting
*
* Note: Requires the separate asset pack (see data/README.md)
*
* Copyright (C) 2016-2017 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

// For reference see http://blog.selfshadow.com/publications/s2013-shading-course/karis/s2013_pbs_epic_notes_v2.pdf

#include "VulkanExampleBase.hpp"
#include "VulkanTexture.hpp"
#include "VulkanModel.hpp"

#include <chrono>

#define ENABLE_VALIDATION false

class VulkanExample : public VulkanExampleBase
{
public:
	bool displaySkybox = true;

	struct Textures {
		vks::TextureCubeMap environmentCube;
		// Generated at runtime
		vks::Texture2D lutBrdf;
		vks::TextureCubeMap irradianceCube;
		vks::TextureCubeMap prefilteredCube;
		// Object texture maps
		vks::Texture2D albedoMap;
		vks::Texture2D normalMap;
		vks::Texture2D aoMap;
		vks::Texture2D metallicMap;
		vks::Texture2D roughnessMap;
	} textures;

	// Vertex layout for the models
	vks::VertexLayout vertexLayout = vks::VertexLayout({
		vks::VERTEX_COMPONENT_POSITION,
		vks::VERTEX_COMPONENT_NORMAL,
		vks::VERTEX_COMPONENT_UV,
	});

	struct Meshes {
		vks::Model skybox;
		vks::Model object;
	} models;

	struct {
		vks::Buffer object;
		vks::Buffer skybox;
		vks::Buffer params;
	} uniformBuffers;

	struct UBOMatrices {
		glm::mat4 projection;
		glm::mat4 model;
		glm::mat4 view;
		glm::vec3 camPos;
	} uboMatrices;

	struct UBOParams {
		glm::vec4 lights[4];
		float exposure = 4.5f;
		float gamma = 2.2f;
	} uboParams;

	struct {
		vk::Pipeline skybox;
		vk::Pipeline pbr;
	} pipelines;

	struct {
		vk::DescriptorSet object;
		vk::DescriptorSet skybox;
	} descriptorSets;

	vk::PipelineLayout pipelineLayout;
	vk::DescriptorSetLayout descriptorSetLayout;

	VulkanExample() : VulkanExampleBase(ENABLE_VALIDATION)
	{
		title = "Vulkan Example - Textured PBR with IBL";

		enableTextOverlay = true;
		camera.type = Camera::CameraType::firstperson;
		camera.movementSpeed = 4.0f;
		camera.setPerspective(60.0f, (float)width / (float)height, 0.1f, 256.0f);
		camera.rotationSpeed = 0.25f;

		camera.setRotation({ -10.75f, 153.0f, 0.0f });
		camera.setPosition({ 1.85f, 0.5f, 5.0f });
	}

	~VulkanExample()
	{
		device.destroyPipeline(pipelines.skybox);
		device.destroyPipeline(pipelines.pbr);

		device.destroyPipelineLayout(pipelineLayout);
		device.destroyDescriptorSetLayout(descriptorSetLayout);

		models.object.destroy();
		models.skybox.destroy();

		uniformBuffers.object.destroy();
		uniformBuffers.skybox.destroy();
		uniformBuffers.params.destroy();
		
		textures.environmentCube.destroy();
		textures.irradianceCube.destroy();
		textures.prefilteredCube.destroy();
		textures.lutBrdf.destroy();
		textures.albedoMap.destroy();
		textures.normalMap.destroy();
		textures.aoMap.destroy();
		textures.metallicMap.destroy();
		textures.roughnessMap.destroy();
	}

	virtual void getEnabledFeatures()
	{
		if (deviceFeatures.samplerAnisotropy) {
			enabledFeatures.samplerAnisotropy = VK_TRUE;
		}
	}

	void buildCommandBuffers()
	{
		vk::CommandBufferBeginInfo cmdBufInfo;

		vk::ClearValue clearValues[2];
		clearValues[0].color = vk::ClearColorValue{ std::array<float, 4>{ 0.1f, 0.1f, 0.1f, 1.0f } };
		clearValues[1].depthStencil = vk::ClearDepthStencilValue{ 1.0f, 0 };

		vk::RenderPassBeginInfo renderPassBeginInfo;
		renderPassBeginInfo.renderPass = renderPass;
		renderPassBeginInfo.renderArea.offset.x = 0;
		renderPassBeginInfo.renderArea.offset.y = 0;
		renderPassBeginInfo.renderArea.extent.width = width;
		renderPassBeginInfo.renderArea.extent.height = height;
		renderPassBeginInfo.clearValueCount = 2;
		renderPassBeginInfo.pClearValues = clearValues;

		for (size_t i = 0; i < drawCmdBuffers.size(); ++i)
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
				drawCmdBuffers[i].bindVertexBuffers(0, models.skybox.vertices.buffer, offsets);
				drawCmdBuffers[i].bindIndexBuffer(models.skybox.indices.buffer, 0, vk::IndexType::eUint32);
				drawCmdBuffers[i].bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.skybox);
				drawCmdBuffers[i].drawIndexed(models.skybox.indexCount, 1, 0, 0, 0);
			}

			// Objects
			drawCmdBuffers[i].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSets.object, nullptr);
			drawCmdBuffers[i].bindVertexBuffers(0, models.object.vertices.buffer, offsets);
			drawCmdBuffers[i].bindIndexBuffer(models.object.indices.buffer, 0, vk::IndexType::eUint32);
			drawCmdBuffers[i].bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.pbr);

			drawCmdBuffers[i].drawIndexed(models.object.indexCount, 1, 0, 0, 0);

			drawCmdBuffers[i].endRenderPass();

			drawCmdBuffers[i].end();
		}
	}

	void loadAssets()
	{
		textures.environmentCube.loadFromFile(ASSET_PATH "textures/hdr/gcanyon_cube.ktx", vk::Format::eR16G16B16A16Sfloat, vulkanDevice, queue);
		models.skybox.loadFromFile(ASSET_PATH "models/cube.obj", vertexLayout, 1.0f, vulkanDevice, queue);
		// PBR model
		models.object.loadFromFile(ASSET_PATH "models/cerberus/cerberus.fbx", vertexLayout, 0.05f, vulkanDevice, queue);
		textures.albedoMap.loadFromFile(ASSET_PATH "models/cerberus/albedo.ktx", vk::Format::eR8G8B8A8Unorm, vulkanDevice, queue);
		textures.normalMap.loadFromFile(ASSET_PATH "models/cerberus/normal.ktx", vk::Format::eR8G8B8A8Unorm, vulkanDevice, queue);
		textures.aoMap.loadFromFile(ASSET_PATH "models/cerberus/ao.ktx", vk::Format::eR8Unorm, vulkanDevice, queue);
		textures.metallicMap.loadFromFile(ASSET_PATH "models/cerberus/metallic.ktx", vk::Format::eR8Unorm, vulkanDevice, queue);
		textures.roughnessMap.loadFromFile(ASSET_PATH "models/cerberus/roughness.ktx", vk::Format::eR8Unorm, vulkanDevice, queue);
	}

	void setupDescriptors()
	{
		// Descriptor Pool
		std::vector<vk::DescriptorPoolSize> poolSizes = {
			vks::initializers::descriptorPoolSize(vk::DescriptorType::eUniformBuffer, 4),
			vks::initializers::descriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, 16)
		};
		vk::DescriptorPoolCreateInfo descriptorPoolInfo =	vks::initializers::descriptorPoolCreateInfo(poolSizes, 2);
		descriptorPool = device.createDescriptorPool(descriptorPoolInfo);

		// Descriptor set layout
		std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings = {
			vks::initializers::descriptorSetLayoutBinding(vk::DescriptorType::eUniformBuffer, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0),
			vks::initializers::descriptorSetLayoutBinding(vk::DescriptorType::eUniformBuffer, vk::ShaderStageFlagBits::eFragment, 1),
			vks::initializers::descriptorSetLayoutBinding(vk::DescriptorType::eCombinedImageSampler, vk::ShaderStageFlagBits::eFragment, 2),
			vks::initializers::descriptorSetLayoutBinding(vk::DescriptorType::eCombinedImageSampler, vk::ShaderStageFlagBits::eFragment, 3),
			vks::initializers::descriptorSetLayoutBinding(vk::DescriptorType::eCombinedImageSampler, vk::ShaderStageFlagBits::eFragment, 4),
			vks::initializers::descriptorSetLayoutBinding(vk::DescriptorType::eCombinedImageSampler, vk::ShaderStageFlagBits::eFragment, 5),
			vks::initializers::descriptorSetLayoutBinding(vk::DescriptorType::eCombinedImageSampler, vk::ShaderStageFlagBits::eFragment, 6),
			vks::initializers::descriptorSetLayoutBinding(vk::DescriptorType::eCombinedImageSampler, vk::ShaderStageFlagBits::eFragment, 7),
			vks::initializers::descriptorSetLayoutBinding(vk::DescriptorType::eCombinedImageSampler, vk::ShaderStageFlagBits::eFragment, 8),
			vks::initializers::descriptorSetLayoutBinding(vk::DescriptorType::eCombinedImageSampler, vk::ShaderStageFlagBits::eFragment, 9),
		};
		vk::DescriptorSetLayoutCreateInfo descriptorLayout = 	vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
		descriptorSetLayout = device.createDescriptorSetLayout(descriptorLayout);

		// Descriptor sets
		vk::DescriptorSetAllocateInfo allocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorPool, &descriptorSetLayout, 1);

		// Objects
		descriptorSets.object = device.allocateDescriptorSets(allocInfo)[0];
		std::vector<vk::WriteDescriptorSet> writeDescriptorSets = {
			vks::initializers::writeDescriptorSet(descriptorSets.object, vk::DescriptorType::eUniformBuffer, 0, &uniformBuffers.object.descriptor),
			vks::initializers::writeDescriptorSet(descriptorSets.object, vk::DescriptorType::eUniformBuffer, 1, &uniformBuffers.params.descriptor),
			vks::initializers::writeDescriptorSet(descriptorSets.object, vk::DescriptorType::eCombinedImageSampler, 2, &textures.irradianceCube.descriptor),
			vks::initializers::writeDescriptorSet(descriptorSets.object, vk::DescriptorType::eCombinedImageSampler, 3, &textures.lutBrdf.descriptor),
			vks::initializers::writeDescriptorSet(descriptorSets.object, vk::DescriptorType::eCombinedImageSampler, 4, &textures.prefilteredCube.descriptor),
			vks::initializers::writeDescriptorSet(descriptorSets.object, vk::DescriptorType::eCombinedImageSampler, 5, &textures.albedoMap.descriptor),
			vks::initializers::writeDescriptorSet(descriptorSets.object, vk::DescriptorType::eCombinedImageSampler, 6, &textures.normalMap.descriptor),
			vks::initializers::writeDescriptorSet(descriptorSets.object, vk::DescriptorType::eCombinedImageSampler, 7, &textures.aoMap.descriptor),
			vks::initializers::writeDescriptorSet(descriptorSets.object, vk::DescriptorType::eCombinedImageSampler, 8, &textures.metallicMap.descriptor),
			vks::initializers::writeDescriptorSet(descriptorSets.object, vk::DescriptorType::eCombinedImageSampler, 9, &textures.roughnessMap.descriptor),
		};
		device.updateDescriptorSets(writeDescriptorSets, nullptr);

		// Sky box
		descriptorSets.skybox = device.allocateDescriptorSets(allocInfo)[0];
		writeDescriptorSets = {
			vks::initializers::writeDescriptorSet(descriptorSets.skybox, vk::DescriptorType::eUniformBuffer, 0, &uniformBuffers.skybox.descriptor),
			vks::initializers::writeDescriptorSet(descriptorSets.skybox, vk::DescriptorType::eUniformBuffer, 1, &uniformBuffers.params.descriptor),
			vks::initializers::writeDescriptorSet(descriptorSets.skybox, vk::DescriptorType::eCombinedImageSampler, 2, &textures.environmentCube.descriptor),
		};
		device.updateDescriptorSets(writeDescriptorSets, nullptr);
	}

	void preparePipelines()
	{
		vk::PipelineInputAssemblyStateCreateInfo inputAssemblyState =
			vks::initializers::pipelineInputAssemblyStateCreateInfo(vk::PrimitiveTopology::eTriangleList, vk::PipelineInputAssemblyStateCreateFlags(), VK_FALSE);

		vk::PipelineRasterizationStateCreateInfo rasterizationState =
			vks::initializers::pipelineRasterizationStateCreateInfo(vk::PolygonMode::eFill, vk::CullModeFlagBits::eBack, vk::FrontFace::eCounterClockwise);

		vk::PipelineColorBlendAttachmentState blendAttachmentState =
			vks::initializers::pipelineColorBlendAttachmentState();

		vk::PipelineColorBlendStateCreateInfo colorBlendState =
			vks::initializers::pipelineColorBlendStateCreateInfo(1, &blendAttachmentState);

		vk::PipelineDepthStencilStateCreateInfo depthStencilState =
			vks::initializers::pipelineDepthStencilStateCreateInfo(VK_FALSE, VK_FALSE, vk::CompareOp::eLessOrEqual);

		vk::PipelineViewportStateCreateInfo viewportState =
			vks::initializers::pipelineViewportStateCreateInfo(1, 1);

		vk::PipelineMultisampleStateCreateInfo multisampleState =
			vks::initializers::pipelineMultisampleStateCreateInfo(vk::SampleCountFlagBits::e1);

		std::vector<vk::DynamicState> dynamicStateEnables = {
			vk::DynamicState::eViewport,
			vk::DynamicState::eScissor
		};
		vk::PipelineDynamicStateCreateInfo dynamicState =
			vks::initializers::pipelineDynamicStateCreateInfo(dynamicStateEnables);

		// Pipeline layout
		vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo = vks::initializers::pipelineLayoutCreateInfo(&descriptorSetLayout, 1);
		pipelineLayout = device.createPipelineLayout(pipelineLayoutCreateInfo);

		// Pipelines
		vk::GraphicsPipelineCreateInfo pipelineCreateInfo = vks::initializers::pipelineCreateInfo(pipelineLayout, renderPass);

		std::array<vk::PipelineShaderStageCreateInfo, 2> shaderStages;

		pipelineCreateInfo.pInputAssemblyState = &inputAssemblyState;
		pipelineCreateInfo.pRasterizationState = &rasterizationState;
		pipelineCreateInfo.pColorBlendState = &colorBlendState;
		pipelineCreateInfo.pMultisampleState = &multisampleState;
		pipelineCreateInfo.pViewportState = &viewportState;
		pipelineCreateInfo.pDepthStencilState = &depthStencilState;
		pipelineCreateInfo.pDynamicState = &dynamicState;
		pipelineCreateInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
		pipelineCreateInfo.pStages = shaderStages.data();

		// Vertex bindings an attributes
		// Binding description
		std::vector<vk::VertexInputBindingDescription> vertexInputBindings = {
			vks::initializers::vertexInputBindingDescription(0, vertexLayout.stride(), vk::VertexInputRate::eVertex),
		};

		// Attribute descriptions
		std::vector<vk::VertexInputAttributeDescription> vertexInputAttributes = {
			vks::initializers::vertexInputAttributeDescription(0, 0, vk::Format::eR32G32B32Sfloat, 0),					// Position
			vks::initializers::vertexInputAttributeDescription(0, 1, vk::Format::eR32G32B32Sfloat, sizeof(float) * 3),	// Normal
			vks::initializers::vertexInputAttributeDescription(0, 2, vk::Format::eR32G32Sfloat, sizeof(float) * 6),		// UV
		};

		vk::PipelineVertexInputStateCreateInfo vertexInputState = vks::initializers::pipelineVertexInputStateCreateInfo();
		vertexInputState.vertexBindingDescriptionCount = static_cast<uint32_t>(vertexInputBindings.size());
		vertexInputState.pVertexBindingDescriptions = vertexInputBindings.data();
		vertexInputState.vertexAttributeDescriptionCount = static_cast<uint32_t>(vertexInputAttributes.size());
		vertexInputState.pVertexAttributeDescriptions = vertexInputAttributes.data();

		pipelineCreateInfo.pVertexInputState = &vertexInputState;

		// Skybox pipeline (background cube)
		shaderStages[0] = loadShader(ASSET_PATH "shaders/pbrtexture/skybox.vert.spv", vk::ShaderStageFlagBits::eVertex);
		shaderStages[1] = loadShader(ASSET_PATH "shaders/pbrtexture/skybox.frag.spv", vk::ShaderStageFlagBits::eFragment);
		pipelines.skybox = device.createGraphicsPipelines(pipelineCache, pipelineCreateInfo)[0];

		// PBR pipeline
		rasterizationState.cullMode = vk::CullModeFlagBits::eFront;
		shaderStages[0] = loadShader(ASSET_PATH "shaders/pbrtexture/pbrtexture.vert.spv", vk::ShaderStageFlagBits::eVertex);
		shaderStages[1] = loadShader(ASSET_PATH "shaders/pbrtexture/pbrtexture.frag.spv", vk::ShaderStageFlagBits::eFragment);
		// Enable depth test and write
		depthStencilState.depthWriteEnable = VK_TRUE;
		depthStencilState.depthTestEnable = VK_TRUE;
		pipelines.pbr = device.createGraphicsPipelines(pipelineCache, pipelineCreateInfo)[0];
	}

	// Generate a BRDF integration map used as a look-up-table (stores roughness / NdotV)
	void generateBRDFLUT()
	{
		auto tStart = std::chrono::high_resolution_clock::now();

		const vk::Format format = vk::Format::eR16G16Sfloat;	// R16G16 is supported pretty much everywhere
		const uint32_t dim = 512;

		// Image
		vk::ImageCreateInfo imageCI;
		imageCI.imageType = vk::ImageType::e2D;
		imageCI.format = format;
		imageCI.extent = vk::Extent3D{ dim, dim, 1 };
		imageCI.mipLevels = 1;
		imageCI.arrayLayers = 1;
		imageCI.samples = vk::SampleCountFlagBits::e1;
		imageCI.tiling = vk::ImageTiling::eOptimal;
		imageCI.usage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled;
		textures.lutBrdf.image = device.createImage(imageCI);
		vk::MemoryAllocateInfo memAlloc;
		vk::MemoryRequirements memReqs;
		memReqs = device.getImageMemoryRequirements(textures.lutBrdf.image);
		memAlloc.allocationSize = memReqs.size;
		memAlloc.memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);
		textures.lutBrdf.deviceMemory = device.allocateMemory(memAlloc);
		device.bindImageMemory(textures.lutBrdf.image, textures.lutBrdf.deviceMemory, 0);
		// Image view
		vk::ImageViewCreateInfo viewCI;
		viewCI.viewType = vk::ImageViewType::e2D;
		viewCI.format = format;
		viewCI.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
		viewCI.subresourceRange.levelCount = 1;
		viewCI.subresourceRange.layerCount = 1;
		viewCI.image = textures.lutBrdf.image;
		textures.lutBrdf.view = device.createImageView(viewCI);
		// Sampler
		vk::SamplerCreateInfo samplerCI = vks::initializers::samplerCreateInfo();
		samplerCI.magFilter = vk::Filter::eLinear;
		samplerCI.minFilter = vk::Filter::eLinear;
		samplerCI.mipmapMode = vk::SamplerMipmapMode::eLinear;
		samplerCI.addressModeU = vk::SamplerAddressMode::eClampToEdge;
		samplerCI.addressModeV = vk::SamplerAddressMode::eClampToEdge;
		samplerCI.addressModeW = vk::SamplerAddressMode::eClampToEdge;
		samplerCI.minLod = 0.0f;
		samplerCI.maxLod = 1.0f;
		samplerCI.borderColor = vk::BorderColor::eFloatOpaqueWhite;
		textures.lutBrdf.sampler = device.createSampler(samplerCI);

		textures.lutBrdf.descriptor.imageView = textures.lutBrdf.view;
		textures.lutBrdf.descriptor.sampler = textures.lutBrdf.sampler;
		textures.lutBrdf.descriptor.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
		textures.lutBrdf.device = vulkanDevice;

		// FB, Att, RP, Pipe, etc.
		vk::AttachmentDescription attDesc = {};
		// Color attachment
		attDesc.format = format;
		attDesc.samples = vk::SampleCountFlagBits::e1;
		attDesc.loadOp = vk::AttachmentLoadOp::eClear;
		attDesc.storeOp = vk::AttachmentStoreOp::eStore;
		attDesc.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
		attDesc.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
		attDesc.initialLayout = vk::ImageLayout::eUndefined;
		attDesc.finalLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
		vk::AttachmentReference colorReference = { 0, vk::ImageLayout::eColorAttachmentOptimal };

		vk::SubpassDescription subpassDescription = {};
		subpassDescription.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
		subpassDescription.colorAttachmentCount = 1;
		subpassDescription.pColorAttachments = &colorReference;

		// Use subpass dependencies for layout transitions
		std::array<vk::SubpassDependency, 2> dependencies;
		dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
		dependencies[0].dstSubpass = 0;
		dependencies[0].srcStageMask = vk::PipelineStageFlagBits::eBottomOfPipe;
		dependencies[0].dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
		dependencies[0].srcAccessMask = vk::AccessFlagBits::eMemoryRead;
		dependencies[0].dstAccessMask = vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite;
		dependencies[0].dependencyFlags = vk::DependencyFlagBits::eByRegion;
		dependencies[1].srcSubpass = 0;
		dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
		dependencies[1].srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
		dependencies[1].dstStageMask = vk::PipelineStageFlagBits::eBottomOfPipe;
		dependencies[1].srcAccessMask = vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite;
		dependencies[1].dstAccessMask = vk::AccessFlagBits::eMemoryRead;
		dependencies[1].dependencyFlags = vk::DependencyFlagBits::eByRegion;

		// Create the actual renderpass
		vk::RenderPassCreateInfo renderPassCI;
		renderPassCI.attachmentCount = 1;
		renderPassCI.pAttachments = &attDesc;
		renderPassCI.subpassCount = 1;
		renderPassCI.pSubpasses = &subpassDescription;
		renderPassCI.dependencyCount = 2;
		renderPassCI.pDependencies = dependencies.data();

		vk::RenderPass renderpass;
		renderpass = device.createRenderPass(renderPassCI);

		vk::FramebufferCreateInfo framebufferCI;
		framebufferCI.renderPass = renderpass;
		framebufferCI.attachmentCount = 1;
		framebufferCI.pAttachments = &textures.lutBrdf.view;
		framebufferCI.width = dim;
		framebufferCI.height = dim;
		framebufferCI.layers = 1;
		
		vk::Framebuffer framebuffer;
		framebuffer = device.createFramebuffer(framebufferCI);

		// Desriptors
		vk::DescriptorSetLayout descriptorsetlayout;
		std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings = {};
		vk::DescriptorSetLayoutCreateInfo descriptorsetlayoutCI = vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
		descriptorsetlayout = device.createDescriptorSetLayout(descriptorsetlayoutCI);

		// Descriptor Pool
		std::vector<vk::DescriptorPoolSize> poolSizes = { vks::initializers::descriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, 1) };
		vk::DescriptorPoolCreateInfo descriptorPoolCI = vks::initializers::descriptorPoolCreateInfo(poolSizes, 2);
		vk::DescriptorPool descriptorpool;
		descriptorpool = device.createDescriptorPool(descriptorPoolCI);

		// Descriptor sets
		vk::DescriptorSet descriptorset;
		vk::DescriptorSetAllocateInfo allocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorpool, &descriptorsetlayout, 1);
		descriptorset = device.allocateDescriptorSets(allocInfo)[0];

		// Pipeline layout
		vk::PipelineLayout pipelinelayout;
		vk::PipelineLayoutCreateInfo pipelineLayoutCI = vks::initializers::pipelineLayoutCreateInfo(&descriptorsetlayout, 1);
		pipelinelayout = device.createPipelineLayout(pipelineLayoutCI);

		// Pipeline
		vk::PipelineInputAssemblyStateCreateInfo inputAssemblyState = vks::initializers::pipelineInputAssemblyStateCreateInfo(vk::PrimitiveTopology::eTriangleList, vk::PipelineInputAssemblyStateCreateFlags(), VK_FALSE);
		vk::PipelineRasterizationStateCreateInfo rasterizationState = vks::initializers::pipelineRasterizationStateCreateInfo(vk::PolygonMode::eFill, vk::CullModeFlagBits::eNone, vk::FrontFace::eCounterClockwise);
		vk::PipelineColorBlendAttachmentState blendAttachmentState = vks::initializers::pipelineColorBlendAttachmentState();
		vk::PipelineColorBlendStateCreateInfo colorBlendState = vks::initializers::pipelineColorBlendStateCreateInfo(1, &blendAttachmentState);
		vk::PipelineDepthStencilStateCreateInfo depthStencilState = vks::initializers::pipelineDepthStencilStateCreateInfo(VK_FALSE, VK_FALSE, vk::CompareOp::eLessOrEqual);
		vk::PipelineViewportStateCreateInfo viewportState = vks::initializers::pipelineViewportStateCreateInfo(1, 1);
		vk::PipelineMultisampleStateCreateInfo multisampleState = vks::initializers::pipelineMultisampleStateCreateInfo(vk::SampleCountFlagBits::e1);
		std::vector<vk::DynamicState> dynamicStateEnables = { vk::DynamicState::eViewport, vk::DynamicState::eScissor };
		vk::PipelineDynamicStateCreateInfo dynamicState = vks::initializers::pipelineDynamicStateCreateInfo(dynamicStateEnables);
		vk::PipelineVertexInputStateCreateInfo emptyInputState = vks::initializers::pipelineVertexInputStateCreateInfo();
		std::array<vk::PipelineShaderStageCreateInfo, 2> shaderStages;

		vk::GraphicsPipelineCreateInfo pipelineCI = vks::initializers::pipelineCreateInfo(pipelinelayout, renderpass);
		pipelineCI.pInputAssemblyState = &inputAssemblyState;
		pipelineCI.pRasterizationState = &rasterizationState;
		pipelineCI.pColorBlendState = &colorBlendState;
		pipelineCI.pMultisampleState = &multisampleState;
		pipelineCI.pViewportState = &viewportState;
		pipelineCI.pDepthStencilState = &depthStencilState;
		pipelineCI.pDynamicState = &dynamicState;
		pipelineCI.stageCount = 2;
		pipelineCI.pStages = shaderStages.data();
		pipelineCI.pVertexInputState = &emptyInputState;

		// Look-up-table (from BRDF) pipeline
		shaderStages[0] = loadShader(ASSET_PATH "shaders/pbrtexture/genbrdflut.vert.spv", vk::ShaderStageFlagBits::eVertex);
		shaderStages[1] = loadShader(ASSET_PATH "shaders/pbrtexture/genbrdflut.frag.spv", vk::ShaderStageFlagBits::eFragment);
		vk::Pipeline pipeline;
		pipeline = device.createGraphicsPipelines(pipelineCache, pipelineCI)[0];

		// Render
		vk::ClearValue clearValues[1];
		clearValues[0].color = vk::ClearColorValue{ std::array<float, 4>{ 0.0f, 0.0f, 0.0f, 1.0f } };

		vk::RenderPassBeginInfo renderPassBeginInfo;
		renderPassBeginInfo.renderPass = renderpass;
		renderPassBeginInfo.renderArea.extent.width = dim;
		renderPassBeginInfo.renderArea.extent.height = dim;
		renderPassBeginInfo.clearValueCount = 1;
		renderPassBeginInfo.pClearValues = clearValues;
		renderPassBeginInfo.framebuffer = framebuffer;

		vk::CommandBuffer cmdBuf = vulkanDevice->createCommandBuffer(vk::CommandBufferLevel::ePrimary, true);
		cmdBuf.beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);
		vk::Viewport viewport = vks::initializers::viewport((float)dim, (float)dim, 0.0f, 1.0f);
		vk::Rect2D scissor = vks::initializers::rect2D(dim, dim, 0, 0);
		cmdBuf.setViewport(0, viewport);
		cmdBuf.setScissor(0, scissor);
		cmdBuf.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);
		vkCmdDraw(cmdBuf, 3, 1, 0, 0);
		cmdBuf.endRenderPass();
		vulkanDevice->flushCommandBuffer(cmdBuf, queue);

		queue.waitIdle();
		
		// todo: cleanup
		device.destroyPipeline(pipeline);
		device.destroyPipelineLayout(pipelinelayout);
		device.destroyRenderPass(renderpass);
		device.destroyFramebuffer(framebuffer);
		device.destroyDescriptorSetLayout(descriptorsetlayout);
		device.destroyDescriptorPool(descriptorpool);

		auto tEnd = std::chrono::high_resolution_clock::now();
		auto tDiff = std::chrono::duration<double, std::milli>(tEnd - tStart).count();
		std::cout << "Generating BRDF LUT took " << tDiff << " ms" << std::endl;
	}

	// Generate an irradiance cube map from the environment cube map
	void generateIrradianceCube()
	{
		auto tStart = std::chrono::high_resolution_clock::now();

		const vk::Format format = vk::Format::eR32G32B32A32Sfloat;
		const int32_t dim = 64;
		const uint32_t numMips = static_cast<uint32_t>(floor(log2(dim))) + 1;

		// Pre-filtered cube map
		// Image
		vk::ImageCreateInfo imageCI;
		imageCI.imageType = vk::ImageType::e2D;
		imageCI.format = format;
		imageCI.extent = vk::Extent3D{ dim, dim, 1 };
		imageCI.mipLevels = numMips;
		imageCI.arrayLayers = 6;
		imageCI.samples = vk::SampleCountFlagBits::e1;
		imageCI.tiling = vk::ImageTiling::eOptimal;
		imageCI.usage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst;
		imageCI.flags = vk::ImageCreateFlagBits::eCubeCompatible;
		textures.irradianceCube.image = device.createImage(imageCI);
		vk::MemoryAllocateInfo memAlloc;
		vk::MemoryRequirements memReqs;
		memReqs = device.getImageMemoryRequirements(textures.irradianceCube.image);
		memAlloc.allocationSize = memReqs.size;
		memAlloc.memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);
		textures.irradianceCube.deviceMemory = device.allocateMemory(memAlloc);
		device.bindImageMemory(textures.irradianceCube.image, textures.irradianceCube.deviceMemory, 0);
		// Image view
		vk::ImageViewCreateInfo viewCI;
		viewCI.viewType = vk::ImageViewType::eCube;
		viewCI.format = format;
		//viewCI.subresourceRange = {};
		viewCI.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
		viewCI.subresourceRange.levelCount = numMips;
		viewCI.subresourceRange.layerCount = 6;
		viewCI.image = textures.irradianceCube.image;
		textures.irradianceCube.view = device.createImageView(viewCI);
		// Sampler
		vk::SamplerCreateInfo samplerCI = vks::initializers::samplerCreateInfo();
		samplerCI.magFilter = vk::Filter::eLinear;
		samplerCI.minFilter = vk::Filter::eLinear;
		samplerCI.mipmapMode = vk::SamplerMipmapMode::eLinear;
		samplerCI.addressModeU = vk::SamplerAddressMode::eClampToEdge;
		samplerCI.addressModeV = vk::SamplerAddressMode::eClampToEdge;
		samplerCI.addressModeW = vk::SamplerAddressMode::eClampToEdge;
		samplerCI.minLod = 0.0f;
		samplerCI.maxLod = static_cast<float>(numMips);
		samplerCI.borderColor = vk::BorderColor::eFloatOpaqueWhite;
		textures.irradianceCube.sampler = device.createSampler(samplerCI);

		textures.irradianceCube.descriptor.imageView = textures.irradianceCube.view;
		textures.irradianceCube.descriptor.sampler = textures.irradianceCube.sampler;
		textures.irradianceCube.descriptor.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
		textures.irradianceCube.device = vulkanDevice;

		// FB, Att, RP, Pipe, etc.
		vk::AttachmentDescription attDesc = {};
		// Color attachment
		attDesc.format = format;
		attDesc.samples = vk::SampleCountFlagBits::e1;
		attDesc.loadOp = vk::AttachmentLoadOp::eClear;
		attDesc.storeOp = vk::AttachmentStoreOp::eStore;
		attDesc.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
		attDesc.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
		attDesc.initialLayout = vk::ImageLayout::eUndefined;
		attDesc.finalLayout = vk::ImageLayout::eColorAttachmentOptimal;
		vk::AttachmentReference colorReference = { 0, vk::ImageLayout::eColorAttachmentOptimal };

		vk::SubpassDescription subpassDescription = {};
		subpassDescription.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
		subpassDescription.colorAttachmentCount = 1;
		subpassDescription.pColorAttachments = &colorReference;

		// Use subpass dependencies for layout transitions
		std::array<vk::SubpassDependency, 2> dependencies;
		dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
		dependencies[0].dstSubpass = 0;
		dependencies[0].srcStageMask = vk::PipelineStageFlagBits::eBottomOfPipe;
		dependencies[0].dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
		dependencies[0].srcAccessMask = vk::AccessFlagBits::eMemoryRead;
		dependencies[0].dstAccessMask = vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite;
		dependencies[0].dependencyFlags = vk::DependencyFlagBits::eByRegion;
		dependencies[1].srcSubpass = 0;
		dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
		dependencies[1].srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
		dependencies[1].dstStageMask = vk::PipelineStageFlagBits::eBottomOfPipe;
		dependencies[1].srcAccessMask = vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite;
		dependencies[1].dstAccessMask = vk::AccessFlagBits::eMemoryRead;
		dependencies[1].dependencyFlags = vk::DependencyFlagBits::eByRegion;

		// Renderpass
		vk::RenderPassCreateInfo renderPassCI;
		renderPassCI.attachmentCount = 1;
		renderPassCI.pAttachments = &attDesc;
		renderPassCI.subpassCount = 1;
		renderPassCI.pSubpasses = &subpassDescription;
		renderPassCI.dependencyCount = 2;
		renderPassCI.pDependencies = dependencies.data();
		vk::RenderPass renderpass;
		renderpass = device.createRenderPass(renderPassCI);

		struct {
			vk::Image image;
			vk::ImageView view;
			vk::DeviceMemory memory;
			vk::Framebuffer framebuffer;
		} offscreen;

		// Offfscreen framebuffer
		{
			// Color attachment
			vk::ImageCreateInfo imageCreateInfo;
			imageCreateInfo.imageType = vk::ImageType::e2D;
			imageCreateInfo.format = format;
			imageCreateInfo.extent = vk::Extent3D{ dim, dim, 1 };
			imageCreateInfo.mipLevels = 1;
			imageCreateInfo.arrayLayers = 1;
			imageCreateInfo.samples = vk::SampleCountFlagBits::e1;
			imageCreateInfo.tiling = vk::ImageTiling::eOptimal;
			imageCreateInfo.initialLayout = vk::ImageLayout::eUndefined;
			imageCreateInfo.usage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferSrc;
			imageCreateInfo.sharingMode = vk::SharingMode::eExclusive;
			offscreen.image = device.createImage(imageCreateInfo);

			vk::MemoryAllocateInfo memAlloc;
			vk::MemoryRequirements memReqs;
			memReqs = device.getImageMemoryRequirements(offscreen.image);
			memAlloc.allocationSize = memReqs.size;
			memAlloc.memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);
			offscreen.memory = device.allocateMemory(memAlloc);
			device.bindImageMemory(offscreen.image, offscreen.memory, 0);

			vk::ImageViewCreateInfo colorImageView;
			colorImageView.viewType = vk::ImageViewType::e2D;
			colorImageView.format = format;
			//colorImageView.flags = 0;
			//colorImageView.subresourceRange = {};
			colorImageView.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
			colorImageView.subresourceRange.baseMipLevel = 0;
			colorImageView.subresourceRange.levelCount = 1;
			colorImageView.subresourceRange.baseArrayLayer = 0;
			colorImageView.subresourceRange.layerCount = 1;
			colorImageView.image = offscreen.image;
			offscreen.view = device.createImageView(colorImageView);

			vk::FramebufferCreateInfo fbufCreateInfo;
			fbufCreateInfo.renderPass = renderpass;
			fbufCreateInfo.attachmentCount = 1;
			fbufCreateInfo.pAttachments = &offscreen.view;
			fbufCreateInfo.width = dim;
			fbufCreateInfo.height = dim;
			fbufCreateInfo.layers = 1;
			offscreen.framebuffer = device.createFramebuffer(fbufCreateInfo);

			vk::CommandBuffer layoutCmd = VulkanExampleBase::createCommandBuffer(vk::CommandBufferLevel::ePrimary, true);
			vks::tools::setImageLayout(
				layoutCmd,
				offscreen.image,
				vk::ImageAspectFlagBits::eColor,
				vk::ImageLayout::eUndefined,
				vk::ImageLayout::eColorAttachmentOptimal);
			VulkanExampleBase::flushCommandBuffer(layoutCmd, queue, true);
		}

		// Descriptors
		vk::DescriptorSetLayout descriptorsetlayout;
		std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings = {
			vks::initializers::descriptorSetLayoutBinding(vk::DescriptorType::eCombinedImageSampler, vk::ShaderStageFlagBits::eFragment, 0),
		};
		vk::DescriptorSetLayoutCreateInfo descriptorsetlayoutCI = vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
		descriptorsetlayout = device.createDescriptorSetLayout(descriptorsetlayoutCI);

		// Descriptor Pool
		std::vector<vk::DescriptorPoolSize> poolSizes = { vks::initializers::descriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, 1) };
		vk::DescriptorPoolCreateInfo descriptorPoolCI = vks::initializers::descriptorPoolCreateInfo(poolSizes, 2);
		vk::DescriptorPool descriptorpool;
		descriptorpool = device.createDescriptorPool(descriptorPoolCI);

		// Descriptor sets
		vk::DescriptorSet descriptorset;
		vk::DescriptorSetAllocateInfo allocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorpool, &descriptorsetlayout, 1);
		descriptorset = device.allocateDescriptorSets(allocInfo)[0];
		vk::WriteDescriptorSet writeDescriptorSet = vks::initializers::writeDescriptorSet(descriptorset, vk::DescriptorType::eCombinedImageSampler, 0, &textures.environmentCube.descriptor);
		device.updateDescriptorSets(writeDescriptorSet, nullptr);

		// Pipeline layout
		struct PushBlock {
			glm::mat4 mvp;
			// Sampling deltas
			float deltaPhi = (2.0f * float(M_PI)) / 180.0f;
			float deltaTheta = (0.5f * float(M_PI)) / 64.0f;
		} pushBlock;

		vk::PipelineLayout pipelinelayout;
		std::vector<vk::PushConstantRange> pushConstantRanges = {
			vks::initializers::pushConstantRange(vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, sizeof(PushBlock), 0),
		};
		vk::PipelineLayoutCreateInfo pipelineLayoutCI = vks::initializers::pipelineLayoutCreateInfo(&descriptorsetlayout, 1);
		pipelineLayoutCI.pushConstantRangeCount = 1;
		pipelineLayoutCI.pPushConstantRanges = pushConstantRanges.data();
		pipelinelayout = device.createPipelineLayout(pipelineLayoutCI);

		// Pipeline
		vk::PipelineInputAssemblyStateCreateInfo inputAssemblyState = vks::initializers::pipelineInputAssemblyStateCreateInfo(vk::PrimitiveTopology::eTriangleList, vk::PipelineInputAssemblyStateCreateFlags(), VK_FALSE);
		vk::PipelineRasterizationStateCreateInfo rasterizationState = vks::initializers::pipelineRasterizationStateCreateInfo(vk::PolygonMode::eFill, vk::CullModeFlagBits::eNone, vk::FrontFace::eCounterClockwise);
		vk::PipelineColorBlendAttachmentState blendAttachmentState = vks::initializers::pipelineColorBlendAttachmentState();
		vk::PipelineColorBlendStateCreateInfo colorBlendState = vks::initializers::pipelineColorBlendStateCreateInfo(1, &blendAttachmentState);
		vk::PipelineDepthStencilStateCreateInfo depthStencilState = vks::initializers::pipelineDepthStencilStateCreateInfo(VK_FALSE, VK_FALSE, vk::CompareOp::eLessOrEqual);
		vk::PipelineViewportStateCreateInfo viewportState = vks::initializers::pipelineViewportStateCreateInfo(1, 1);
		vk::PipelineMultisampleStateCreateInfo multisampleState = vks::initializers::pipelineMultisampleStateCreateInfo(vk::SampleCountFlagBits::e1);
		std::vector<vk::DynamicState> dynamicStateEnables = { vk::DynamicState::eViewport, vk::DynamicState::eScissor };
		vk::PipelineDynamicStateCreateInfo dynamicState = vks::initializers::pipelineDynamicStateCreateInfo(dynamicStateEnables);
		// Vertex input state
		vk::VertexInputBindingDescription vertexInputBinding = vks::initializers::vertexInputBindingDescription(0, vertexLayout.stride(), vk::VertexInputRate::eVertex);
		vk::VertexInputAttributeDescription vertexInputAttribute = vks::initializers::vertexInputAttributeDescription(0, 0, vk::Format::eR32G32B32Sfloat, 0);

		vk::PipelineVertexInputStateCreateInfo vertexInputState = vks::initializers::pipelineVertexInputStateCreateInfo();
		vertexInputState.vertexBindingDescriptionCount = 1;
		vertexInputState.pVertexBindingDescriptions = &vertexInputBinding;
		vertexInputState.vertexAttributeDescriptionCount = 1;
		vertexInputState.pVertexAttributeDescriptions = &vertexInputAttribute;

		std::array<vk::PipelineShaderStageCreateInfo, 2> shaderStages;

		vk::GraphicsPipelineCreateInfo pipelineCI = vks::initializers::pipelineCreateInfo(pipelinelayout, renderpass);
		pipelineCI.pInputAssemblyState = &inputAssemblyState;
		pipelineCI.pRasterizationState = &rasterizationState;
		pipelineCI.pColorBlendState = &colorBlendState;
		pipelineCI.pMultisampleState = &multisampleState;
		pipelineCI.pViewportState = &viewportState;
		pipelineCI.pDepthStencilState = &depthStencilState;
		pipelineCI.pDynamicState = &dynamicState;
		pipelineCI.stageCount = 2;
		pipelineCI.pStages = shaderStages.data();
		pipelineCI.pVertexInputState = &vertexInputState;
		pipelineCI.renderPass = renderpass;

		shaderStages[0] = loadShader(ASSET_PATH "shaders/pbrtexture/filtercube.vert.spv", vk::ShaderStageFlagBits::eVertex);
		shaderStages[1] = loadShader(ASSET_PATH "shaders/pbrtexture/irradiancecube.frag.spv", vk::ShaderStageFlagBits::eFragment);
		vk::Pipeline pipeline;
		pipeline = device.createGraphicsPipelines(pipelineCache, pipelineCI)[0];

		// Render

		vk::ClearValue clearValues[1];
		clearValues[0].color = vk::ClearColorValue{ std::array<float, 4>{ 0.0f, 0.0f, 0.2f, 0.0f } };

		vk::RenderPassBeginInfo renderPassBeginInfo;
		// Reuse render pass from example pass
		renderPassBeginInfo.renderPass = renderpass;
		renderPassBeginInfo.framebuffer = offscreen.framebuffer;
		renderPassBeginInfo.renderArea.extent.width = dim;
		renderPassBeginInfo.renderArea.extent.height = dim;
		renderPassBeginInfo.clearValueCount = 1;
		renderPassBeginInfo.pClearValues = clearValues;

		std::vector<glm::mat4> matrices = {
			// POSITIVE_X
			glm::rotate(glm::rotate(glm::mat4(), glm::radians(90.0f), glm::vec3(0.0f, 1.0f, 0.0f)), glm::radians(180.0f), glm::vec3(1.0f, 0.0f, 0.0f)),
			// NEGATIVE_X
			glm::rotate(glm::rotate(glm::mat4(), glm::radians(-90.0f), glm::vec3(0.0f, 1.0f, 0.0f)), glm::radians(180.0f), glm::vec3(1.0f, 0.0f, 0.0f)),
			// POSITIVE_Y
			glm::rotate(glm::mat4(), glm::radians(-90.0f), glm::vec3(1.0f, 0.0f, 0.0f)),
			// NEGATIVE_Y
			glm::rotate(glm::mat4(), glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f)),
			// POSITIVE_Z
			glm::rotate(glm::mat4(), glm::radians(180.0f), glm::vec3(1.0f, 0.0f, 0.0f)),
			// NEGATIVE_Z
			glm::rotate(glm::mat4(), glm::radians(180.0f), glm::vec3(0.0f, 0.0f, 1.0f)),
		};

		vk::CommandBuffer cmdBuf = vulkanDevice->createCommandBuffer(vk::CommandBufferLevel::ePrimary, true);

		vk::Viewport viewport = vks::initializers::viewport((float)dim, (float)dim, 0.0f, 1.0f);
		vk::Rect2D scissor = vks::initializers::rect2D(dim, dim, 0, 0);

		cmdBuf.setViewport(0, viewport);
		cmdBuf.setScissor(0, scissor);

		vk::ImageSubresourceRange subresourceRange;
		subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
		subresourceRange.baseMipLevel = 0;
		subresourceRange.levelCount = numMips;
		subresourceRange.layerCount = 6;

		// Change image layout for all cubemap faces to transfer destination
		vks::tools::setImageLayout(
			cmdBuf,
			textures.irradianceCube.image,
			vk::ImageLayout::eUndefined,
			vk::ImageLayout::eTransferDstOptimal,
			subresourceRange);

		for (uint32_t m = 0; m < numMips; m++) {
			for (uint32_t f = 0; f < 6; f++) {
				viewport.width = static_cast<float>(dim * std::pow(0.5f, m));
				viewport.height = static_cast<float>(dim * std::pow(0.5f, m));
				cmdBuf.setViewport(0, viewport);

				// Render scene from cube face's point of view
				cmdBuf.beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);

				// Update shader push constant block
				pushBlock.mvp = glm::perspective((float)(M_PI / 2.0), 1.0f, 0.1f, 512.0f) * matrices[f];

				cmdBuf.pushConstants(pipelinelayout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, sizeof(PushBlock), &pushBlock);

				cmdBuf.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);
				cmdBuf.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelinelayout, 0, descriptorset, nullptr);

				std::vector<vk::DeviceSize> offsets = { 0 };

				cmdBuf.bindVertexBuffers(0, models.skybox.vertices.buffer, offsets);
				cmdBuf.bindIndexBuffer(models.skybox.indices.buffer, 0, vk::IndexType::eUint32);
				cmdBuf.drawIndexed(models.skybox.indexCount, 1, 0, 0, 0);

				cmdBuf.endRenderPass();

				vks::tools::setImageLayout(
					cmdBuf,
					offscreen.image,
					vk::ImageAspectFlagBits::eColor,
					vk::ImageLayout::eColorAttachmentOptimal,
					vk::ImageLayout::eTransferSrcOptimal);

				// Copy region for transfer from framebuffer to cube face
				vk::ImageCopy copyRegion = {};

				copyRegion.srcSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
				copyRegion.srcSubresource.baseArrayLayer = 0;
				copyRegion.srcSubresource.mipLevel = 0;
				copyRegion.srcSubresource.layerCount = 1;
				copyRegion.srcOffset = vk::Offset3D{ 0, 0, 0 };

				copyRegion.dstSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
				copyRegion.dstSubresource.baseArrayLayer = f;
				copyRegion.dstSubresource.mipLevel = m;
				copyRegion.dstSubresource.layerCount = 1;
				copyRegion.dstOffset = vk::Offset3D{ 0, 0, 0 };

				copyRegion.extent.width = static_cast<uint32_t>(viewport.width);
				copyRegion.extent.height = static_cast<uint32_t>(viewport.height);
				copyRegion.extent.depth = 1;

				cmdBuf.copyImage(
					offscreen.image,
					vk::ImageLayout::eTransferSrcOptimal,
					textures.irradianceCube.image,
					vk::ImageLayout::eTransferDstOptimal,
					copyRegion);

				// Transform framebuffer color attachment back 
				vks::tools::setImageLayout(
					cmdBuf,
					offscreen.image,
					vk::ImageAspectFlagBits::eColor,
					vk::ImageLayout::eTransferSrcOptimal,
					vk::ImageLayout::eColorAttachmentOptimal);
			}
		}

		vks::tools::setImageLayout(
			cmdBuf,
			textures.irradianceCube.image,
			vk::ImageLayout::eTransferDstOptimal,
			vk::ImageLayout::eShaderReadOnlyOptimal,
			subresourceRange);

		vulkanDevice->flushCommandBuffer(cmdBuf, queue);

		// todo: cleanup
		device.destroyRenderPass(renderpass);
		device.destroyFramebuffer(offscreen.framebuffer);
		device.freeMemory(offscreen.memory);
		device.destroyImageView(offscreen.view);
		device.destroyImage(offscreen.image);
		device.destroyDescriptorPool(descriptorpool);
		device.destroyDescriptorSetLayout(descriptorsetlayout);
		device.destroyPipeline(pipeline);
		device.destroyPipelineLayout(pipelinelayout);

		auto tEnd = std::chrono::high_resolution_clock::now();
		auto tDiff = std::chrono::duration<double, std::milli>(tEnd - tStart).count();
		std::cout << "Generating irradiance cube with " << numMips << " mip levels took " << tDiff << " ms" << std::endl;
	}

	// Prefilter environment cubemap
	// See https://placeholderart.wordpress.com/2015/07/28/implementation-notes-runtime-environment-map-filtering-for-image-based-lighting/
	void generatePrefilteredCube()
	{
		auto tStart = std::chrono::high_resolution_clock::now();

		const vk::Format format = vk::Format::eR16G16B16A16Sfloat;
		const int32_t dim = 512;
		const uint32_t numMips = static_cast<uint32_t>(floor(log2(dim))) + 1;

		// Pre-filtered cube map
		// Image
		vk::ImageCreateInfo imageCI;
		imageCI.imageType = vk::ImageType::e2D;
		imageCI.format = format;
		imageCI.extent = vk::Extent3D{ dim, dim, 1 };
		imageCI.mipLevels = numMips;
		imageCI.arrayLayers = 6;
		imageCI.samples = vk::SampleCountFlagBits::e1;
		imageCI.tiling = vk::ImageTiling::eOptimal;
		imageCI.usage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst;
		imageCI.flags = vk::ImageCreateFlagBits::eCubeCompatible;
		textures.prefilteredCube.image = device.createImage(imageCI);
		vk::MemoryAllocateInfo memAlloc;
		vk::MemoryRequirements memReqs;
		memReqs = device.getImageMemoryRequirements(textures.prefilteredCube.image);
		memAlloc.allocationSize = memReqs.size;
		memAlloc.memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);
		textures.prefilteredCube.deviceMemory = device.allocateMemory(memAlloc);
		device.bindImageMemory(textures.prefilteredCube.image, textures.prefilteredCube.deviceMemory, 0);
		// Image view
		vk::ImageViewCreateInfo viewCI;
		viewCI.viewType = vk::ImageViewType::eCube;
		viewCI.format = format;
		//viewCI.subresourceRange = {};
		viewCI.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
		viewCI.subresourceRange.levelCount = numMips;
		viewCI.subresourceRange.layerCount = 6;
		viewCI.image = textures.prefilteredCube.image;
		textures.prefilteredCube.view = device.createImageView(viewCI);
		// Sampler
		vk::SamplerCreateInfo samplerCI = vks::initializers::samplerCreateInfo();
		samplerCI.magFilter = vk::Filter::eLinear;
		samplerCI.minFilter = vk::Filter::eLinear;
		samplerCI.mipmapMode = vk::SamplerMipmapMode::eLinear;
		samplerCI.addressModeU = vk::SamplerAddressMode::eClampToEdge;
		samplerCI.addressModeV = vk::SamplerAddressMode::eClampToEdge;
		samplerCI.addressModeW = vk::SamplerAddressMode::eClampToEdge;
		samplerCI.minLod = 0.0f;
		samplerCI.maxLod = static_cast<float>(numMips);
		samplerCI.borderColor = vk::BorderColor::eFloatOpaqueWhite;
		textures.prefilteredCube.sampler = device.createSampler(samplerCI);

		textures.prefilteredCube.descriptor.imageView = textures.prefilteredCube.view;
		textures.prefilteredCube.descriptor.sampler = textures.prefilteredCube.sampler;
		textures.prefilteredCube.descriptor.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
		textures.prefilteredCube.device = vulkanDevice;

		// FB, Att, RP, Pipe, etc.
		vk::AttachmentDescription attDesc = {};
		// Color attachment
		attDesc.format = format;
		attDesc.samples = vk::SampleCountFlagBits::e1;
		attDesc.loadOp = vk::AttachmentLoadOp::eClear;
		attDesc.storeOp = vk::AttachmentStoreOp::eStore;
		attDesc.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
		attDesc.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
		attDesc.initialLayout = vk::ImageLayout::eUndefined;
		attDesc.finalLayout = vk::ImageLayout::eColorAttachmentOptimal;
		vk::AttachmentReference colorReference = { 0, vk::ImageLayout::eColorAttachmentOptimal };

		vk::SubpassDescription subpassDescription = {};
		subpassDescription.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
		subpassDescription.colorAttachmentCount = 1;
		subpassDescription.pColorAttachments = &colorReference;

		// Use subpass dependencies for layout transitions
		std::array<vk::SubpassDependency, 2> dependencies;
		dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
		dependencies[0].dstSubpass = 0;
		dependencies[0].srcStageMask = vk::PipelineStageFlagBits::eBottomOfPipe;
		dependencies[0].dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
		dependencies[0].srcAccessMask = vk::AccessFlagBits::eMemoryRead;
		dependencies[0].dstAccessMask = vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite;
		dependencies[0].dependencyFlags = vk::DependencyFlagBits::eByRegion;
		dependencies[1].srcSubpass = 0;
		dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
		dependencies[1].srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
		dependencies[1].dstStageMask = vk::PipelineStageFlagBits::eBottomOfPipe;
		dependencies[1].srcAccessMask = vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite;
		dependencies[1].dstAccessMask = vk::AccessFlagBits::eMemoryRead;
		dependencies[1].dependencyFlags = vk::DependencyFlagBits::eByRegion;

		// Renderpass
		vk::RenderPassCreateInfo renderPassCI;
		renderPassCI.attachmentCount = 1;
		renderPassCI.pAttachments = &attDesc;
		renderPassCI.subpassCount = 1;
		renderPassCI.pSubpasses = &subpassDescription;
		renderPassCI.dependencyCount = 2;
		renderPassCI.pDependencies = dependencies.data();
		vk::RenderPass renderpass;
		renderpass = device.createRenderPass(renderPassCI);

		struct {
			vk::Image image;
			vk::ImageView view;
			vk::DeviceMemory memory;
			vk::Framebuffer framebuffer;
		} offscreen;

		// Offfscreen framebuffer
		{
			// Color attachment
			vk::ImageCreateInfo imageCreateInfo;
			imageCreateInfo.imageType = vk::ImageType::e2D;
			imageCreateInfo.format = format;
			imageCreateInfo.extent = vk::Extent3D{ dim, dim, 1 };
			imageCreateInfo.mipLevels = 1;
			imageCreateInfo.arrayLayers = 1;
			imageCreateInfo.samples = vk::SampleCountFlagBits::e1;
			imageCreateInfo.tiling = vk::ImageTiling::eOptimal;
			imageCreateInfo.initialLayout = vk::ImageLayout::eUndefined;
			imageCreateInfo.usage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferSrc;
			imageCreateInfo.sharingMode = vk::SharingMode::eExclusive;
			offscreen.image = device.createImage(imageCreateInfo);

			vk::MemoryAllocateInfo memAlloc;
			vk::MemoryRequirements memReqs;
			memReqs = device.getImageMemoryRequirements(offscreen.image);
			memAlloc.allocationSize = memReqs.size;
			memAlloc.memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);
			offscreen.memory = device.allocateMemory(memAlloc);
			device.bindImageMemory(offscreen.image, offscreen.memory, 0);

			vk::ImageViewCreateInfo colorImageView;
			colorImageView.viewType = vk::ImageViewType::e2D;
			colorImageView.format = format;
			//colorImageView.flags = 0;
			//colorImageView.subresourceRange = {};
			colorImageView.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
			colorImageView.subresourceRange.baseMipLevel = 0;
			colorImageView.subresourceRange.levelCount = 1;
			colorImageView.subresourceRange.baseArrayLayer = 0;
			colorImageView.subresourceRange.layerCount = 1;
			colorImageView.image = offscreen.image;
			offscreen.view = device.createImageView(colorImageView);

			vk::FramebufferCreateInfo fbufCreateInfo;
			fbufCreateInfo.renderPass = renderpass;
			fbufCreateInfo.attachmentCount = 1;
			fbufCreateInfo.pAttachments = &offscreen.view;
			fbufCreateInfo.width = dim;
			fbufCreateInfo.height = dim;
			fbufCreateInfo.layers = 1;
			offscreen.framebuffer = device.createFramebuffer(fbufCreateInfo);

			vk::CommandBuffer layoutCmd = VulkanExampleBase::createCommandBuffer(vk::CommandBufferLevel::ePrimary, true);
			vks::tools::setImageLayout(
				layoutCmd,
				offscreen.image,
				vk::ImageAspectFlagBits::eColor,
				vk::ImageLayout::eUndefined,
				vk::ImageLayout::eColorAttachmentOptimal);
			VulkanExampleBase::flushCommandBuffer(layoutCmd, queue, true);
		}

		// Descriptors
		vk::DescriptorSetLayout descriptorsetlayout;
		std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings = {
			vks::initializers::descriptorSetLayoutBinding(vk::DescriptorType::eCombinedImageSampler, vk::ShaderStageFlagBits::eFragment, 0),
		};
		vk::DescriptorSetLayoutCreateInfo descriptorsetlayoutCI = vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
		descriptorsetlayout = device.createDescriptorSetLayout(descriptorsetlayoutCI);

		// Descriptor Pool
		std::vector<vk::DescriptorPoolSize> poolSizes = { vks::initializers::descriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, 1) };
		vk::DescriptorPoolCreateInfo descriptorPoolCI = vks::initializers::descriptorPoolCreateInfo(poolSizes, 2);
		vk::DescriptorPool descriptorpool;
		descriptorpool = device.createDescriptorPool(descriptorPoolCI);

		// Descriptor sets
		vk::DescriptorSet descriptorset;
		vk::DescriptorSetAllocateInfo allocInfo =	vks::initializers::descriptorSetAllocateInfo(descriptorpool, &descriptorsetlayout, 1);
		descriptorset = device.allocateDescriptorSets(allocInfo)[0];
		vk::WriteDescriptorSet writeDescriptorSet = vks::initializers::writeDescriptorSet(descriptorset, vk::DescriptorType::eCombinedImageSampler, 0, &textures.environmentCube.descriptor);
		device.updateDescriptorSets(writeDescriptorSet, nullptr);

		// Pipeline layout
		struct PushBlock {
			glm::mat4 mvp;
			float roughness;
			uint32_t numSamples = 32u;
		} pushBlock;

		vk::PipelineLayout pipelinelayout;
		std::vector<vk::PushConstantRange> pushConstantRanges = {
			vks::initializers::pushConstantRange(vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, sizeof(PushBlock), 0),
		};
		vk::PipelineLayoutCreateInfo pipelineLayoutCI = vks::initializers::pipelineLayoutCreateInfo(&descriptorsetlayout, 1);
		pipelineLayoutCI.pushConstantRangeCount = 1;
		pipelineLayoutCI.pPushConstantRanges = pushConstantRanges.data();
		pipelinelayout = device.createPipelineLayout(pipelineLayoutCI);

		// Pipeline
		vk::PipelineInputAssemblyStateCreateInfo inputAssemblyState = vks::initializers::pipelineInputAssemblyStateCreateInfo(vk::PrimitiveTopology::eTriangleList, vk::PipelineInputAssemblyStateCreateFlags(), VK_FALSE);
		vk::PipelineRasterizationStateCreateInfo rasterizationState = vks::initializers::pipelineRasterizationStateCreateInfo(vk::PolygonMode::eFill, vk::CullModeFlagBits::eNone, vk::FrontFace::eCounterClockwise);
		vk::PipelineColorBlendAttachmentState blendAttachmentState = vks::initializers::pipelineColorBlendAttachmentState();
		vk::PipelineColorBlendStateCreateInfo colorBlendState = vks::initializers::pipelineColorBlendStateCreateInfo(1, &blendAttachmentState);
		vk::PipelineDepthStencilStateCreateInfo depthStencilState = vks::initializers::pipelineDepthStencilStateCreateInfo(VK_FALSE, VK_FALSE, vk::CompareOp::eLessOrEqual);
		vk::PipelineViewportStateCreateInfo viewportState = vks::initializers::pipelineViewportStateCreateInfo(1, 1);
		vk::PipelineMultisampleStateCreateInfo multisampleState = vks::initializers::pipelineMultisampleStateCreateInfo(vk::SampleCountFlagBits::e1);
		std::vector<vk::DynamicState> dynamicStateEnables = { vk::DynamicState::eViewport, vk::DynamicState::eScissor };
		vk::PipelineDynamicStateCreateInfo dynamicState = vks::initializers::pipelineDynamicStateCreateInfo(dynamicStateEnables);
		// Vertex input state
		vk::VertexInputBindingDescription vertexInputBinding = vks::initializers::vertexInputBindingDescription(0, vertexLayout.stride(), vk::VertexInputRate::eVertex);
		vk::VertexInputAttributeDescription vertexInputAttribute = vks::initializers::vertexInputAttributeDescription(0, 0, vk::Format::eR32G32B32Sfloat, 0);

		vk::PipelineVertexInputStateCreateInfo vertexInputState = vks::initializers::pipelineVertexInputStateCreateInfo();
		vertexInputState.vertexBindingDescriptionCount = 1;
		vertexInputState.pVertexBindingDescriptions = &vertexInputBinding;
		vertexInputState.vertexAttributeDescriptionCount = 1;
		vertexInputState.pVertexAttributeDescriptions = &vertexInputAttribute;

		std::array<vk::PipelineShaderStageCreateInfo, 2> shaderStages;

		vk::GraphicsPipelineCreateInfo pipelineCI = vks::initializers::pipelineCreateInfo(pipelinelayout, renderpass);
		pipelineCI.pInputAssemblyState = &inputAssemblyState;
		pipelineCI.pRasterizationState = &rasterizationState;
		pipelineCI.pColorBlendState = &colorBlendState;
		pipelineCI.pMultisampleState = &multisampleState;
		pipelineCI.pViewportState = &viewportState;
		pipelineCI.pDepthStencilState = &depthStencilState;
		pipelineCI.pDynamicState = &dynamicState;
		pipelineCI.stageCount = 2;
		pipelineCI.pStages = shaderStages.data();
		pipelineCI.pVertexInputState = &vertexInputState;
		pipelineCI.renderPass = renderpass;

		shaderStages[0] = loadShader(ASSET_PATH "shaders/pbrtexture/filtercube.vert.spv", vk::ShaderStageFlagBits::eVertex);
		shaderStages[1] = loadShader(ASSET_PATH "shaders/pbrtexture/prefilterenvmap.frag.spv", vk::ShaderStageFlagBits::eFragment);
		vk::Pipeline pipeline;
		pipeline = device.createGraphicsPipelines(pipelineCache, pipelineCI)[0];

		// Render

		vk::ClearValue clearValues[1];
		clearValues[0].color = vk::ClearColorValue{ std::array<float, 4>{ 0.0f, 0.0f, 0.2f, 0.0f } };

		vk::RenderPassBeginInfo renderPassBeginInfo;
		// Reuse render pass from example pass
		renderPassBeginInfo.renderPass = renderpass;
		renderPassBeginInfo.framebuffer = offscreen.framebuffer;
		renderPassBeginInfo.renderArea.extent.width = dim;
		renderPassBeginInfo.renderArea.extent.height = dim;
		renderPassBeginInfo.clearValueCount = 1;
		renderPassBeginInfo.pClearValues = clearValues;

		std::vector<glm::mat4> matrices = {
			// POSITIVE_X
			glm::rotate(glm::rotate(glm::mat4(), glm::radians(90.0f), glm::vec3(0.0f, 1.0f, 0.0f)), glm::radians(180.0f), glm::vec3(1.0f, 0.0f, 0.0f)),
			// NEGATIVE_X
			glm::rotate(glm::rotate(glm::mat4(), glm::radians(-90.0f), glm::vec3(0.0f, 1.0f, 0.0f)), glm::radians(180.0f), glm::vec3(1.0f, 0.0f, 0.0f)),
			// POSITIVE_Y
			glm::rotate(glm::mat4(), glm::radians(-90.0f), glm::vec3(1.0f, 0.0f, 0.0f)),
			// NEGATIVE_Y
			glm::rotate(glm::mat4(), glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f)),
			// POSITIVE_Z
			glm::rotate(glm::mat4(), glm::radians(180.0f), glm::vec3(1.0f, 0.0f, 0.0f)),
			// NEGATIVE_Z
			glm::rotate(glm::mat4(), glm::radians(180.0f), glm::vec3(0.0f, 0.0f, 1.0f)),
		};

		vk::CommandBuffer cmdBuf = vulkanDevice->createCommandBuffer(vk::CommandBufferLevel::ePrimary, true);

		vk::Viewport viewport = vks::initializers::viewport((float)dim, (float)dim, 0.0f, 1.0f);
		vk::Rect2D scissor = vks::initializers::rect2D(dim, dim, 0, 0);
		
		cmdBuf.setViewport(0, viewport);
		cmdBuf.setScissor(0, scissor);

		vk::ImageSubresourceRange subresourceRange;
		subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
		subresourceRange.baseMipLevel = 0;
		subresourceRange.levelCount = numMips;
		subresourceRange.layerCount = 6;

		// Change image layout for all cubemap faces to transfer destination
		vks::tools::setImageLayout(
			cmdBuf,
			textures.prefilteredCube.image,
			vk::ImageLayout::eUndefined,
			vk::ImageLayout::eTransferDstOptimal,
			subresourceRange);

		for (uint32_t m = 0; m < numMips; m++) {
			pushBlock.roughness = (float)m / (float)(numMips - 1);
			for (uint32_t f = 0; f < 6; f++) {
				viewport.width = static_cast<float>(dim * std::pow(0.5f, m));
				viewport.height = static_cast<float>(dim * std::pow(0.5f, m));
				cmdBuf.setViewport(0, viewport);

				// Render scene from cube face's point of view
				cmdBuf.beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);

				// Update shader push constant block
				pushBlock.mvp = glm::perspective((float)(M_PI / 2.0), 1.0f, 0.1f, 512.0f) * matrices[f];

				cmdBuf.pushConstants(pipelinelayout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, sizeof(PushBlock), &pushBlock);

				cmdBuf.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);
				cmdBuf.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelinelayout, 0, descriptorset, nullptr);

				std::vector<vk::DeviceSize> offsets = { 0 };

				cmdBuf.bindVertexBuffers(0, models.skybox.vertices.buffer, offsets);
				cmdBuf.bindIndexBuffer(models.skybox.indices.buffer, 0, vk::IndexType::eUint32);
				cmdBuf.drawIndexed(models.skybox.indexCount, 1, 0, 0, 0);

				cmdBuf.endRenderPass();

				vks::tools::setImageLayout(
					cmdBuf, 
					offscreen.image, 
					vk::ImageAspectFlagBits::eColor, 
					vk::ImageLayout::eColorAttachmentOptimal, 
					vk::ImageLayout::eTransferSrcOptimal);

				// Copy region for transfer from framebuffer to cube face
				vk::ImageCopy copyRegion = {};

				copyRegion.srcSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
				copyRegion.srcSubresource.baseArrayLayer = 0;
				copyRegion.srcSubresource.mipLevel = 0;
				copyRegion.srcSubresource.layerCount = 1;
				copyRegion.srcOffset = vk::Offset3D{ 0, 0, 0 };

				copyRegion.dstSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
				copyRegion.dstSubresource.baseArrayLayer = f;
				copyRegion.dstSubresource.mipLevel = m;
				copyRegion.dstSubresource.layerCount = 1;
				copyRegion.dstOffset = vk::Offset3D{ 0, 0, 0 };

				copyRegion.extent.width = static_cast<uint32_t>(viewport.width);
				copyRegion.extent.height = static_cast<uint32_t>(viewport.height);
				copyRegion.extent.depth = 1;

				cmdBuf.copyImage(
					offscreen.image,
					vk::ImageLayout::eTransferSrcOptimal,
					textures.prefilteredCube.image,
					vk::ImageLayout::eTransferDstOptimal,
					copyRegion);

				// Transform framebuffer color attachment back 
				vks::tools::setImageLayout(
					cmdBuf,
					offscreen.image,
					vk::ImageAspectFlagBits::eColor,
					vk::ImageLayout::eTransferSrcOptimal,
					vk::ImageLayout::eColorAttachmentOptimal);
			}
		}

		vks::tools::setImageLayout(
			cmdBuf,
			textures.prefilteredCube.image,
			vk::ImageLayout::eTransferDstOptimal,
			vk::ImageLayout::eShaderReadOnlyOptimal,
			subresourceRange);

		vulkanDevice->flushCommandBuffer(cmdBuf, queue);

		// todo: cleanup
		device.destroyRenderPass(renderpass);
		device.destroyFramebuffer(offscreen.framebuffer);
		device.freeMemory(offscreen.memory);
		device.destroyImageView(offscreen.view);
		device.destroyImage(offscreen.image);
		device.destroyDescriptorPool(descriptorpool);
		device.destroyDescriptorSetLayout(descriptorsetlayout);
		device.destroyPipeline(pipeline);
		device.destroyPipelineLayout(pipelinelayout);

		auto tEnd = std::chrono::high_resolution_clock::now();
		auto tDiff = std::chrono::duration<double, std::milli>(tEnd - tStart).count();
		std::cout << "Generating pre-filtered enivornment cube with " << numMips << " mip levels took " << tDiff << " ms" << std::endl;
	}

	// Prepare and initialize uniform buffer containing shader uniforms
	void prepareUniformBuffers()
	{
		// Objact vertex shader uniform buffer
		vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eUniformBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			&uniformBuffers.object,
			sizeof(uboMatrices));

		// Skybox vertex shader uniform buffer
		vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eUniformBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			&uniformBuffers.skybox,
			sizeof(uboMatrices));

		// Shared parameter uniform buffer
		vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eUniformBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			&uniformBuffers.params,
			sizeof(uboParams));

		// Map persistent
		uniformBuffers.object.map();
		uniformBuffers.skybox.map();
		uniformBuffers.params.map();

		updateUniformBuffers();
		updateParams();
	}

	void updateUniformBuffers()
	{
		// 3D object
		uboMatrices.projection = camera.matrices.perspective;
		uboMatrices.view = camera.matrices.view;
		uboMatrices.model = glm::rotate(glm::mat4(), glm::radians(-90.0f), glm::vec3(0.0f, 1.0f, 0.0f));
		uboMatrices.camPos = camera.position * -1.0f;
		memcpy(uniformBuffers.object.mapped, &uboMatrices, sizeof(uboMatrices));

		// Skybox
		uboMatrices.model = glm::mat4(glm::mat3(camera.matrices.view));
		memcpy(uniformBuffers.skybox.mapped, &uboMatrices, sizeof(uboMatrices));
	}

	void updateParams()
	{
		const float p = 15.0f;
		uboParams.lights[0] = glm::vec4(-p, -p*0.5f, -p, 1.0f);
		uboParams.lights[1] = glm::vec4(-p, -p*0.5f,  p, 1.0f);
		uboParams.lights[2] = glm::vec4( p, -p*0.5f,  p, 1.0f);
		uboParams.lights[3] = glm::vec4( p, -p*0.5f, -p, 1.0f);

		memcpy(uniformBuffers.params.mapped, &uboParams, sizeof(uboParams));
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
		generateBRDFLUT();
		generateIrradianceCube();
		generatePrefilteredCube();
		prepareUniformBuffers();
		setupDescriptors();
		preparePipelines();
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
		updateTextOverlay();
	}

	void toggleSkyBox()
	{
		displaySkybox = !displaySkybox;
		buildCommandBuffers();
	}

	void changeExposure(float delta)
	{
		uboParams.exposure += delta;
		if (uboParams.exposure < 0.01f) {
			uboParams.exposure = 0.01f;
		}
		updateParams();
		updateTextOverlay();
	}

	void changeGamma(float delta)
	{
		uboParams.gamma += delta;
		if (uboParams.gamma < 0.01f) {
			uboParams.gamma = 0.01f;
		}
		updateParams();
		updateTextOverlay();
	}

	virtual void keyPressed(uint32_t keyCode)
	{
		switch (keyCode)
		{
		case KEY_F2:
		case GAMEPAD_BUTTON_A:
			toggleSkyBox();
			break;
		case KEY_KPADD:
		case GAMEPAD_BUTTON_R1:
			changeExposure(0.1f);
			break;
		case KEY_KPSUB:
		case GAMEPAD_BUTTON_L1:
			changeExposure(-0.1f);
			break;
		case KEY_F3:
			changeGamma(-0.1f);
			break;
		case KEY_F4:
			changeGamma(0.1f);
			break;
		}
	}

	virtual void getOverlayText(VulkanTextOverlay *textOverlay)
	{
#if defined(__ANDROID__)
#else
		textOverlay->addText("Exposure: " + std::to_string(uboParams.exposure) + " (-/+)", 5.0f, 85.0f, VulkanTextOverlay::alignLeft);
		textOverlay->addText("Gamma: " + std::to_string(uboParams.gamma) + " (F3/F4)", 5.0f, 100.0f, VulkanTextOverlay::alignLeft);
#endif
	}
};

VULKAN_EXAMPLE_MAIN()