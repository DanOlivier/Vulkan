/*
* Vulkan Example - Compute shader image processing
*
* Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include "VulkanExampleBase.hpp"
#include "VulkanTexture.hpp"
#include "keycodes.hpp"

#define VERTEX_BUFFER_BIND_ID 0
#define ENABLE_VALIDATION false

// Vertex layout for this example
struct Vertex {
	float pos[3];
	float uv[2];
};

class VulkanExample : public VulkanExampleBase
{
private:
	vks::Texture2D textureColorMap;
	vks::Texture2D textureComputeTarget;
public:
	struct {
		vk::PipelineVertexInputStateCreateInfo inputState;
		std::vector<vk::VertexInputBindingDescription> bindingDescriptions;
		std::vector<vk::VertexInputAttributeDescription> attributeDescriptions;
	} vertices;

	// Resources for the graphics part of the example
	struct {
		vk::DescriptorSetLayout descriptorSetLayout;	// Image display shader binding layout
		vk::DescriptorSet descriptorSetPreCompute;	// Image display shader bindings before compute shader image manipulation
		vk::DescriptorSet descriptorSetPostCompute;	// Image display shader bindings after compute shader image manipulation
		vk::Pipeline pipeline;						// Image display pipeline
		vk::PipelineLayout pipelineLayout;			// Layout of the graphics pipeline
	} graphics;

	// Resources for the compute part of the example
	struct Compute {
		vk::Queue queue;								// Separate queue for compute commands (queue family may differ from the one used for graphics)
		vk::CommandPool commandPool;					// Use a separate command pool (queue family may differ from the one used for graphics)
		vk::CommandBuffer commandBuffer;				// Command buffer storing the dispatch commands and barriers
		vk::Fence fence;								// Synchronization fence to avoid rewriting compute CB if still in use
		vk::DescriptorSetLayout descriptorSetLayout;	// Compute shader binding layout
		vk::DescriptorSet descriptorSet;				// Compute shader bindings
		vk::PipelineLayout pipelineLayout;			// Layout of the compute pipeline
		std::vector<vk::Pipeline> pipelines;			// Compute pipelines for image filters
		uint32_t pipelineIndex = 0;					// Current image filtering compute pipeline index
		uint32_t queueFamilyIndex;					// Family index of the graphics queue, used for barriers
	} compute;

	vks::Buffer vertexBuffer;
	vks::Buffer indexBuffer;
	uint32_t indexCount;

	vks::Buffer uniformBufferVS;

	struct {
		glm::mat4 projection;
		glm::mat4 model;
	} uboVS;

	struct {
	} pipelines;

	int vertexBufferSize;

	VulkanExample() : VulkanExampleBase(ENABLE_VALIDATION)
	{
		zoom = -2.0f;
		enableTextOverlay = true;
		title = "Vulkan Example - Compute shader image processing";
	}

	~VulkanExample()
	{
		// Graphics
		device.destroyPipeline(graphics.pipeline);
		device.destroyPipelineLayout(graphics.pipelineLayout);
		device.destroyDescriptorSetLayout(graphics.descriptorSetLayout);

		// Compute
		for (auto& pipeline : compute.pipelines)
		{
			device.destroyPipeline(pipeline);
		}
		device.destroyPipelineLayout(compute.pipelineLayout);
		device.destroyDescriptorSetLayout(compute.descriptorSetLayout);
		device.destroyFence(compute.fence);
		device.destroyCommandPool(compute.commandPool);

		vertexBuffer.destroy();
		indexBuffer.destroy();
		uniformBufferVS.destroy();

		textureColorMap.destroy();
		textureComputeTarget.destroy();
	}

	// Prepare a texture target that is used to store compute shader calculations
	void prepareTextureTarget(vks::Texture *tex, uint32_t width, uint32_t height, vk::Format format)
	{
		vk::FormatProperties formatProperties;

		// Get device properties for the requested texture format
		formatProperties = physicalDevice.getFormatProperties(format);
		// Check if requested image format supports image storage operations
		assert(formatProperties.optimalTilingFeatures & vk::FormatFeatureFlagBits::eStorageImage);

		// Prepare blit target texture
		tex->width = width;
		tex->height = height;

		vk::ImageCreateInfo imageCreateInfo;
		imageCreateInfo.imageType = vk::ImageType::e2D;
		imageCreateInfo.format = format;
		imageCreateInfo.extent = vk::Extent3D{ width, height, 1 };
		imageCreateInfo.mipLevels = 1;
		imageCreateInfo.arrayLayers = 1;
		imageCreateInfo.samples = vk::SampleCountFlagBits::e1;
		imageCreateInfo.tiling = vk::ImageTiling::eOptimal;
		// Image will be sampled in the fragment shader and used as storage target in the compute shader
		imageCreateInfo.usage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage;
		//imageCreateInfo.flags = 0;
		// Sharing mode exclusive means that ownership of the image does not need to be explicitly transferred between the compute and graphics queue
		imageCreateInfo.sharingMode = vk::SharingMode::eExclusive;

		vk::MemoryAllocateInfo memAllocInfo;
		vk::MemoryRequirements memReqs;

		tex->image = device.createImage(imageCreateInfo);

		memReqs = device.getImageMemoryRequirements(tex->image);
		memAllocInfo.allocationSize = memReqs.size;
		memAllocInfo.memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);
		tex->deviceMemory = device.allocateMemory(memAllocInfo);
		device.bindImageMemory(tex->image, tex->deviceMemory, 0);

		vk::CommandBuffer layoutCmd = VulkanExampleBase::createCommandBuffer(vk::CommandBufferLevel::ePrimary, true);

		tex->imageLayout = vk::ImageLayout::eGeneral;
		vks::tools::setImageLayout(
			layoutCmd, tex->image, 
			vk::ImageAspectFlagBits::eColor, 
			vk::ImageLayout::eUndefined, 
			tex->imageLayout);

		VulkanExampleBase::flushCommandBuffer(layoutCmd, queue, true);

		// Create sampler
		vk::SamplerCreateInfo sampler;
		sampler.magFilter = vk::Filter::eLinear;
		sampler.minFilter = vk::Filter::eLinear;
		sampler.mipmapMode = vk::SamplerMipmapMode::eLinear;
		sampler.addressModeU = vk::SamplerAddressMode::eClampToBorder;
		sampler.addressModeV = sampler.addressModeU;
		sampler.addressModeW = sampler.addressModeU;
		sampler.mipLodBias = 0.0f;
		sampler.maxAnisotropy = 1.0f;
		sampler.compareOp = vk::CompareOp::eNever;
		sampler.minLod = 0.0f;
		sampler.maxLod = 0.0f;
		sampler.borderColor = vk::BorderColor::eFloatOpaqueWhite;
		tex->sampler = device.createSampler(sampler);

		// Create image view
		vk::ImageViewCreateInfo view;
		view.viewType = vk::ImageViewType::e2D;
		view.format = format;
		view.components = { vk::ComponentSwizzle::eR, vk::ComponentSwizzle::eG, vk::ComponentSwizzle::eB, vk::ComponentSwizzle::eA };
		view.subresourceRange = vk::ImageSubresourceRange{ vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 };
		view.image = tex->image;
		tex->view = device.createImageView(view);

		// Initialize a descriptor for later use
		tex->descriptor.imageLayout = tex->imageLayout;
		tex->descriptor.imageView = tex->view;
		tex->descriptor.sampler = tex->sampler;
		tex->device = vulkanDevice;
	}

	void loadAssets()
	{
		textureColorMap.loadFromFile(
			getAssetPath() + "textures/het_kanonschot_rgba8.ktx",
			vk::Format::eR8G8B8A8Unorm,
			vulkanDevice,
			queue,
			vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage,
			vk::ImageLayout::eGeneral);
	}

	void buildCommandBuffers()
	{
		// Destroy command buffers if already present
		if (!checkCommandBuffers())
		{
			destroyCommandBuffers();
			createCommandBuffers();
		}

		vk::CommandBufferBeginInfo cmdBufInfo;

		vk::ClearValue clearValues[2];
		clearValues[0].color = defaultClearColor;
		clearValues[1].depthStencil = vk::ClearDepthStencilValue{ 1.0f, 0 };

		vk::RenderPassBeginInfo renderPassBeginInfo;
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

			// Image memory barrier to make sure that compute shader writes are finished before sampling from the texture
			vk::ImageMemoryBarrier imageMemoryBarrier = {};

			// We won't be changing the layout of the image
			imageMemoryBarrier.oldLayout = vk::ImageLayout::eGeneral;
			imageMemoryBarrier.newLayout = vk::ImageLayout::eGeneral;
			imageMemoryBarrier.image = textureComputeTarget.image;
			imageMemoryBarrier.subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 };
			imageMemoryBarrier.srcAccessMask = vk::AccessFlagBits::eShaderWrite;
			imageMemoryBarrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;
			drawCmdBuffers[i].pipelineBarrier(
				vk::PipelineStageFlagBits::eComputeShader,
				vk::PipelineStageFlagBits::eFragmentShader,
				vk::DependencyFlags(),
				nullptr,
				nullptr,
				imageMemoryBarrier);
			drawCmdBuffers[i].beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);

			vk::Viewport viewport = vks::initializers::viewport((float)width * 0.5f, (float)height, 0.0f, 1.0f);
			drawCmdBuffers[i].setViewport(0, viewport);

			vk::Rect2D scissor = vks::initializers::rect2D(width, height, 0, 0);
			drawCmdBuffers[i].setScissor(0, scissor);

			std::vector<vk::DeviceSize> offsets = { 0 };
			drawCmdBuffers[i].bindVertexBuffers(VERTEX_BUFFER_BIND_ID, vertexBuffer.buffer, offsets);
			drawCmdBuffers[i].bindIndexBuffer(indexBuffer.buffer, 0, vk::IndexType::eUint32);

			// Left (pre compute)
			drawCmdBuffers[i].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, graphics.pipelineLayout, 0, graphics.descriptorSetPreCompute, nullptr);
			drawCmdBuffers[i].bindPipeline(vk::PipelineBindPoint::eGraphics, graphics.pipeline);

			drawCmdBuffers[i].drawIndexed(indexCount, 1, 0, 0, 0);

			// Right (post compute)
			drawCmdBuffers[i].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, graphics.pipelineLayout, 0, graphics.descriptorSetPostCompute, nullptr);
			drawCmdBuffers[i].bindPipeline(vk::PipelineBindPoint::eGraphics, graphics.pipeline);

			viewport.x = (float)width / 2.0f;
			drawCmdBuffers[i].setViewport(0, viewport);
			drawCmdBuffers[i].drawIndexed(indexCount, 1, 0, 0, 0);

			drawCmdBuffers[i].endRenderPass();

			drawCmdBuffers[i].end();
		}

	}

	void buildComputeCommandBuffer()
	{
		// Flush the queue if we're rebuilding the command buffer after a pipeline change to ensure it's not currently in use
		compute.queue.waitIdle();

		vk::CommandBufferBeginInfo cmdBufInfo;

		compute.commandBuffer.begin(cmdBufInfo);

		compute.commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, compute.pipelines[compute.pipelineIndex]);
		compute.commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, compute.pipelineLayout, 0, compute.descriptorSet, nullptr);

		vkCmdDispatch(compute.commandBuffer, textureComputeTarget.width / 16, textureComputeTarget.height / 16, 1);

		vkEndCommandBuffer(compute.commandBuffer);
	}

	// Setup vertices for a single uv-mapped quad
	void generateQuad()
	{
		// Setup vertices for a single uv-mapped quad made from two triangles
		std::vector<Vertex> vertices =
		{
			{ {  1.0f,  1.0f, 0.0f }, { 1.0f, 1.0f } },
			{ { -1.0f,  1.0f, 0.0f }, { 0.0f, 1.0f } },
			{ { -1.0f, -1.0f, 0.0f }, { 0.0f, 0.0f } },
			{ {  1.0f, -1.0f, 0.0f }, { 1.0f, 0.0f } }
		};

		// Setup indices
		std::vector<uint32_t> indices = { 0,1,2, 2,3,0 };
		indexCount = static_cast<uint32_t>(indices.size());

		// Create buffers
		// For the sake of simplicity we won't stage the vertex data to the gpu memory
		// Vertex buffer
		vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eVertexBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			&vertexBuffer,
			vertices.size() * sizeof(Vertex),
			vertices.data());
		// Index buffer
		vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eIndexBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			&indexBuffer,
			indices.size() * sizeof(uint32_t),
			indices.data());
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
		vertices.attributeDescriptions.resize(2);
		// Location 0 : Position
		vertices.attributeDescriptions[0] =
			vks::initializers::vertexInputAttributeDescription(
				VERTEX_BUFFER_BIND_ID,
				0,
				vk::Format::eR32G32B32Sfloat,
				offsetof(Vertex, pos));
		// Location 1 : Texture coordinates
		vertices.attributeDescriptions[1] =
			vks::initializers::vertexInputAttributeDescription(
				VERTEX_BUFFER_BIND_ID,
				1,
				vk::Format::eR32G32Sfloat,
				offsetof(Vertex, uv));

		// Assign to vertex buffer
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
			// Graphics pipeline uses image samplers for display
			vks::initializers::descriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, 2),
			// Compute pipeline uses a sampled image for reading
			vks::initializers::descriptorPoolSize(vk::DescriptorType::eSampledImage, 1),
			// Compute pipelines uses a storage image for image reads and writes
			vks::initializers::descriptorPoolSize(vk::DescriptorType::eStorageImage, 2),
		};

		vk::DescriptorPoolCreateInfo descriptorPoolInfo =
			vks::initializers::descriptorPoolCreateInfo(
				poolSizes.size(),
				poolSizes.data(),
				3);

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

		graphics.descriptorSetLayout = device.createDescriptorSetLayout(descriptorLayout);
		
		vk::PipelineLayoutCreateInfo pPipelineLayoutCreateInfo =
			vks::initializers::pipelineLayoutCreateInfo(
				&graphics.descriptorSetLayout,
				1);

		graphics.pipelineLayout = device.createPipelineLayout(pPipelineLayoutCreateInfo);
	}

	void setupDescriptorSet()
	{
		vk::DescriptorSetAllocateInfo allocInfo =
			vks::initializers::descriptorSetAllocateInfo(
				descriptorPool,
				&graphics.descriptorSetLayout,
				1);

		graphics.descriptorSetPostCompute = device.allocateDescriptorSets(allocInfo)[0];

		std::vector<vk::WriteDescriptorSet> writeDescriptorSets =
		{
			// Binding 0 : Vertex shader uniform buffer
			vks::initializers::writeDescriptorSet(
				graphics.descriptorSetPostCompute,
				vk::DescriptorType::eUniformBuffer,
				0,
				&uniformBufferVS.descriptor),
			// Binding 1 : Fragment shader texture sampler
			vks::initializers::writeDescriptorSet(
				graphics.descriptorSetPostCompute,
				vk::DescriptorType::eCombinedImageSampler,
				1,
				&textureComputeTarget.descriptor)
		};

		device.updateDescriptorSets(writeDescriptorSets, nullptr);

		// Base image (before compute post process)
		allocInfo =
			vks::initializers::descriptorSetAllocateInfo(
				descriptorPool,
				&graphics.descriptorSetLayout,
				1);

		graphics.descriptorSetPreCompute = device.allocateDescriptorSets(allocInfo)[0];

		std::vector<vk::WriteDescriptorSet> baseImageWriteDescriptorSets =
		{
			// Binding 0 : Vertex shader uniform buffer
			vks::initializers::writeDescriptorSet(
				graphics.descriptorSetPreCompute,
				vk::DescriptorType::eUniformBuffer,
				0,
				&uniformBufferVS.descriptor),
			// Binding 1 : Fragment shader texture sampler
			vks::initializers::writeDescriptorSet(
				graphics.descriptorSetPreCompute,
				vk::DescriptorType::eCombinedImageSampler,
				1,
				&textureColorMap.descriptor)
		};

		device.updateDescriptorSets(baseImageWriteDescriptorSets, nullptr);
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
				vk::CullModeFlagBits::eNone,
				vk::FrontFace::eCounterClockwise);

		vk::PipelineColorBlendAttachmentState blendAttachmentState =
			vks::initializers::pipelineColorBlendAttachmentState();

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

		// Rendering pipeline
		// Load shaders
		std::array<vk::PipelineShaderStageCreateInfo,2> shaderStages;

		shaderStages[0] = loadShader(getAssetPath() + "shaders/texture.vert.spv", vk::ShaderStageFlagBits::eVertex);
		shaderStages[1] = loadShader(getAssetPath() + "shaders/texture.frag.spv", vk::ShaderStageFlagBits::eFragment);

		vk::GraphicsPipelineCreateInfo pipelineCreateInfo =
			vks::initializers::pipelineCreateInfo(
				graphics.pipelineLayout,
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
		pipelineCreateInfo.renderPass = renderPass;

		graphics.pipeline = device.createGraphicsPipelines(pipelineCache, pipelineCreateInfo)[0];
	}

	// Find and create a compute capable device queue
	void getComputeQueue()
	{
		std::vector<vk::QueueFamilyProperties> queueFamilyProperties = 
			physicalDevice.getQueueFamilyProperties();

		// Some devices have dedicated compute queues, so we first try to find a queue that supports compute and not graphics
		bool computeQueueFound = false;
		for (uint32_t i = 0; i < queueFamilyProperties.size(); i++)
		{
			if ((queueFamilyProperties[i].queueFlags & vk::QueueFlagBits::eCompute) && 
				(~(queueFamilyProperties[i].queueFlags & vk::QueueFlagBits::eGraphics)))
			{
				compute.queueFamilyIndex = i;
				computeQueueFound = true;
				break;
			}
		}
		// If there is no dedicated compute queue, just find the first queue family that supports compute
		if (!computeQueueFound)
		{
			for (uint32_t i = 0; i < static_cast<uint32_t>(queueFamilyProperties.size()); i++)
			{
				if (queueFamilyProperties[i].queueFlags & vk::QueueFlagBits::eCompute)
				{
					compute.queueFamilyIndex = i;
					computeQueueFound = true;
					break;
				}
			}
		}

		// Compute is mandatory in Vulkan, so there must be at least one queue family that supports compute
		assert(computeQueueFound);
		// Get a compute queue from the device
		compute.queue = device.getQueue(compute.queueFamilyIndex, 0);
	}

	void prepareCompute()
	{
		getComputeQueue();

		// Create compute pipeline
		// Compute pipelines are created separate from graphics pipelines even if they use the same queue

		std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings = {
			// Binding 0 : Sampled image (read)
			vks::initializers::descriptorSetLayoutBinding(
				vk::DescriptorType::eStorageImage,
				vk::ShaderStageFlagBits::eCompute,
				0),
			// Binding 1 : Sampled image (write)
			vks::initializers::descriptorSetLayoutBinding(
				vk::DescriptorType::eStorageImage,
				vk::ShaderStageFlagBits::eCompute,
				1),
		};

		vk::DescriptorSetLayoutCreateInfo descriptorLayout =
			vks::initializers::descriptorSetLayoutCreateInfo(
				setLayoutBindings.data(),
				setLayoutBindings.size());

		compute.descriptorSetLayout = device.createDescriptorSetLayout(descriptorLayout);

		vk::PipelineLayoutCreateInfo pPipelineLayoutCreateInfo =
			vks::initializers::pipelineLayoutCreateInfo(
				&compute.descriptorSetLayout,
				1);

		compute.pipelineLayout = device.createPipelineLayout(pPipelineLayoutCreateInfo);

		vk::DescriptorSetAllocateInfo allocInfo =
			vks::initializers::descriptorSetAllocateInfo(
				descriptorPool,
				&compute.descriptorSetLayout,
				1);

		compute.descriptorSet = device.allocateDescriptorSets(allocInfo)[0];

		std::vector<vk::WriteDescriptorSet> computeWriteDescriptorSets =
		{
			// Binding 0 : Sampled image (read)
			vks::initializers::writeDescriptorSet(
				compute.descriptorSet,
				vk::DescriptorType::eStorageImage,
				0,
				&textureColorMap.descriptor),
			// Binding 1 : Sampled image (write)
			vks::initializers::writeDescriptorSet(
				compute.descriptorSet,
				vk::DescriptorType::eStorageImage,
				1,
				&textureComputeTarget.descriptor)
		};

		device.updateDescriptorSets(computeWriteDescriptorSets, nullptr);


		// Create compute shader pipelines
		vk::ComputePipelineCreateInfo computePipelineCreateInfo =
			vks::initializers::computePipelineCreateInfo(
				compute.pipelineLayout);

		// One pipeline for each effect
		std::vector<std::string> shaderNames = { "sharpen", "edgedetect", "emboss" };
		for (auto& shaderName : shaderNames)
		{
			std::string fileName = getAssetPath() + "shaders/" + shaderName + ".comp.spv";
			computePipelineCreateInfo.stage = loadShader(fileName.c_str(), vk::ShaderStageFlagBits::eCompute);
			vk::Pipeline pipeline;
			pipeline = device.createComputePipelines(pipelineCache, computePipelineCreateInfo)[0];

			compute.pipelines.push_back(pipeline);
		}

		// Separate command pool as queue family for compute may be different than graphics
		vk::CommandPoolCreateInfo cmdPoolInfo = {};

		cmdPoolInfo.queueFamilyIndex = compute.queueFamilyIndex;
		cmdPoolInfo.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
		compute.commandPool = device.createCommandPool(cmdPoolInfo);

		// Create a command buffer for compute operations
		vk::CommandBufferAllocateInfo cmdBufAllocateInfo =
			vks::initializers::commandBufferAllocateInfo(
				compute.commandPool,
				vk::CommandBufferLevel::ePrimary,
				1);

		compute.commandBuffer = device.allocateCommandBuffers(cmdBufAllocateInfo)[0];

		// Fence for compute CB sync
		vk::FenceCreateInfo fenceCreateInfo = vks::initializers::fenceCreateInfo(vk::FenceCreateFlagBits::eSignaled);
		compute.fence = device.createFence(fenceCreateInfo);

		// Build a single command buffer containing the compute dispatch commands
		buildComputeCommandBuffer();
	}

	// Prepare and initialize uniform buffer containing shader uniforms
	void prepareUniformBuffers()
	{
		// Vertex shader uniform buffer block
		vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eUniformBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			&uniformBufferVS,
			sizeof(uboVS));

		// Map persistent
		uniformBufferVS.map();

		updateUniformBuffers();
	}

	void updateUniformBuffers()
	{
		// Vertex shader uniform buffer block
		uboVS.projection = glm::perspective(glm::radians(60.0f), (float)width*0.5f / (float)height, 0.1f, 256.0f);
		glm::mat4 viewMatrix = glm::translate(glm::mat4(), glm::vec3(0.0f, 0.0f, zoom));

		uboVS.model = viewMatrix * glm::translate(glm::mat4(), cameraPos);
		uboVS.model = glm::rotate(uboVS.model, glm::radians(rotation.x), glm::vec3(1.0f, 0.0f, 0.0f));
		uboVS.model = glm::rotate(uboVS.model, glm::radians(rotation.y), glm::vec3(0.0f, 1.0f, 0.0f));
		uboVS.model = glm::rotate(uboVS.model, glm::radians(rotation.z), glm::vec3(0.0f, 0.0f, 1.0f));

		memcpy(uniformBufferVS.mapped, &uboVS, sizeof(uboVS));
	}

	void draw()
	{
		VulkanExampleBase::prepareFrame();

		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &drawCmdBuffers[currentBuffer];
		queue.submit(submitInfo, vk::Fence(nullptr));

		VulkanExampleBase::submitFrame();

		// Submit compute commands
		// Use a fence to ensure that compute command buffer has finished executin before using it again
		device.waitForFences(compute.fence, VK_TRUE, UINT64_MAX);
		device.resetFences(compute.fence);

		vk::SubmitInfo computeSubmitInfo;
		computeSubmitInfo.commandBufferCount = 1;
		computeSubmitInfo.pCommandBuffers = &compute.commandBuffer;

		compute.queue.submit(computeSubmitInfo, compute.fence);
	}

	void prepare()
	{
		VulkanExampleBase::prepare();
		loadAssets();
		generateQuad();
		setupVertexDescriptions();
		prepareUniformBuffers();
		prepareTextureTarget(&textureComputeTarget, textureColorMap.width, textureColorMap.height, vk::Format::eR8G8B8A8Unorm);
		setupDescriptorSetLayout();
		preparePipelines();
		setupDescriptorPool();
		setupDescriptorSet();
		prepareCompute();
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

	virtual void switchComputePipeline(int32_t dir)
	{
		if ((dir < 0) && (compute.pipelineIndex > 0))
		{
			compute.pipelineIndex--;
			buildComputeCommandBuffer();
		}
		if ((dir > 0) && (compute.pipelineIndex < compute.pipelines.size() - 1))
		{
			compute.pipelineIndex++;
			buildComputeCommandBuffer();
		}
	}

	virtual void keyPressed(uint32_t keyCode)
	{
		switch (keyCode)
		{
		case KEY_KPADD:
		case GAMEPAD_BUTTON_R1:
			switchComputePipeline(1);
			break;
		case KEY_KPSUB:
		case GAMEPAD_BUTTON_L1:
			switchComputePipeline(-1);
			break;
		}
	}

	virtual void getOverlayText(VulkanTextOverlay *textOverlay)
	{
#if defined(__ANDROID__)
		textOverlay->addText("Press \"L1/R1\" to change shaders", 5.0f, 85.0f, VulkanTextOverlay::alignLeft);
#else
		textOverlay->addText("Press \"NUMPAD +/-\" to change shaders", 5.0f, 85.0f, VulkanTextOverlay::alignLeft);
#endif
	}
};

VULKAN_EXAMPLE_MAIN()
