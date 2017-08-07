/*
* Vulkan Example - Compute shader ray tracing
*
* Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <vector>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <vulkan/vulkan.hpp>
#include "vulkanexamplebase.h"
#include "VulkanTexture.hpp"

#define VERTEX_BUFFER_BIND_ID 0
#define ENABLE_VALIDATION false

#if defined(__ANDROID__)
#define TEX_DIM 1024
#else
#define TEX_DIM 2048
#endif

class VulkanExample : public VulkanExampleBase
{
public:
	vks::Texture textureComputeTarget;

	// Resources for the graphics part of the example
	struct {
		vk::DescriptorSetLayout descriptorSetLayout;	// Raytraced image display shader binding layout
		vk::DescriptorSet descriptorSetPreCompute;	// Raytraced image display shader bindings before compute shader image manipulation
		vk::DescriptorSet descriptorSet;				// Raytraced image display shader bindings after compute shader image manipulation
		vk::Pipeline pipeline;						// Raytraced image display pipeline
		vk::PipelineLayout pipelineLayout;			// Layout of the graphics pipeline
	} graphics;

	// Resources for the compute part of the example
	struct {
		struct {
			vks::Buffer spheres;						// (Shader) storage buffer object with scene spheres
			vks::Buffer planes;						// (Shader) storage buffer object with scene planes
		} storageBuffers;
		vks::Buffer uniformBuffer;					// Uniform buffer object containing scene data
		vk::Queue queue;								// Separate queue for compute commands (queue family may differ from the one used for graphics)
		vk::CommandPool commandPool;					// Use a separate command pool (queue family may differ from the one used for graphics)
		vk::CommandBuffer commandBuffer;				// Command buffer storing the dispatch commands and barriers
		vk::Fence fence;								// Synchronization fence to avoid rewriting compute CB if still in use
		vk::DescriptorSetLayout descriptorSetLayout;	// Compute shader binding layout
		vk::DescriptorSet descriptorSet;				// Compute shader bindings
		vk::PipelineLayout pipelineLayout;			// Layout of the compute pipeline
		vk::Pipeline pipeline;						// Compute raytracing pipeline
		struct UBOCompute {							// Compute shader uniform block object
			glm::vec3 lightPos;
			float aspectRatio;						// Aspect ratio of the viewport
			glm::vec4 fogColor = glm::vec4(0.0f);
			struct {
				glm::vec3 pos = glm::vec3(0.0f, 0.0f, 4.0f);
				glm::vec3 lookat = glm::vec3(0.0f, 0.5f, 0.0f);
				float fov = 10.0f;
			} camera;
		} ubo;
	} compute;

	// SSBO sphere declaration 
	struct Sphere {									// Shader uses std140 layout (so we only use vec4 instead of vec3)
		glm::vec3 pos;								
		float radius;
		glm::vec3 diffuse;
		float specular;
		uint32_t id;								// Id used to identify sphere for raytracing
		glm::ivec3 _pad;
	};

	// SSBO plane declaration
	struct Plane {
		glm::vec3 normal;
		float distance;
		glm::vec3 diffuse;
		float specular;
		uint32_t id;
		glm::ivec3 _pad;
	};

	VulkanExample() : VulkanExampleBase(ENABLE_VALIDATION)
	{
		title = "Vulkan Example - Compute shader ray tracing";
		enableTextOverlay = true;
		compute.ubo.aspectRatio = (float)width / (float)height;
		timerSpeed *= 0.25f;

		camera.type = Camera::CameraType::lookat;
		camera.setPerspective(60.0f, (float)width / (float)height, 0.1f, 512.0f);
		camera.setRotation(glm::vec3(0.0f, 0.0f, 0.0f));
		camera.setTranslation(glm::vec3(0.0f, 0.0f, -4.0f));
		camera.rotationSpeed = 0.0f;
		camera.movementSpeed = 2.5f;
	}

	~VulkanExample()
	{
		// Graphics
		device.destroyPipeline(graphics.pipeline);
		device.destroyPipelineLayout(graphics.pipelineLayout);
		device.destroyDescriptorSetLayout(graphics.descriptorSetLayout);

		// Compute
		device.destroyPipeline(compute.pipeline);
		device.destroyPipelineLayout(compute.pipelineLayout);
		device.destroyDescriptorSetLayout(compute.descriptorSetLayout);
		device.destroyFence(compute.fence);
		device.destroyCommandPool(compute.commandPool);
		compute.uniformBuffer.destroy();
		compute.storageBuffers.spheres.destroy();
		compute.storageBuffers.planes.destroy();

		textureComputeTarget.destroy();
	}

	// Prepare a texture target that is used to store compute shader calculations
	void prepareTextureTarget(vks::Texture *tex, uint32_t width, uint32_t height, vk::Format format)
	{
		// Get device properties for the requested texture format
		vk::FormatProperties formatProperties;
		formatProperties = physicalDevice.getFormatProperties(format);
		// Check if requested image format supports image storage operations
		assert(formatProperties.optimalTilingFeatures & vk::FormatFeatureFlagBits::eStorageImage);

		// Prepare blit target texture
		tex->width = width;
		tex->height = height;

		vk::ImageCreateInfo imageCreateInfo = vks::initializers::imageCreateInfo();
		imageCreateInfo.imageType = vk::ImageType::e2D;
		imageCreateInfo.format = format;
		imageCreateInfo.extent = vk::Extent3D{ width, height, 1 };
		imageCreateInfo.mipLevels = 1;
		imageCreateInfo.arrayLayers = 1;
		imageCreateInfo.samples = vk::SampleCountFlagBits::e1;
		imageCreateInfo.tiling = vk::ImageTiling::eOptimal;
		imageCreateInfo.initialLayout = vk::ImageLayout::eUndefined;
		// Image will be sampled in the fragment shader and used as storage target in the compute shader
		imageCreateInfo.usage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage;
		//imageCreateInfo.flags = 0;

		vk::MemoryAllocateInfo memAllocInfo = vks::initializers::memoryAllocateInfo();
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
			layoutCmd, 
			tex->image,
			vk::ImageAspectFlagBits::eColor, 
			vk::ImageLayout::eUndefined,
			tex->imageLayout);

		VulkanExampleBase::flushCommandBuffer(layoutCmd, queue, true);

		// Create sampler
		vk::SamplerCreateInfo sampler = vks::initializers::samplerCreateInfo();
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
		vk::ImageViewCreateInfo view = vks::initializers::imageViewCreateInfo();
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

	void buildCommandBuffers()
	{
		// Destroy command buffers if already present
		if (!checkCommandBuffers())
		{
			destroyCommandBuffers();
			createCommandBuffers();
		}

		vk::CommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();

		vk::ClearValue clearValues[2];
		clearValues[0].color = defaultClearColor;
		clearValues[0].color = vk::ClearColorValue{ std::array<float, 4>{0.0f, 0.0f, 0.2f, 0.0f} };
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

			// Image memory barrier to make sure that compute shader writes are finished before sampling from the texture
			vk::ImageMemoryBarrier imageMemoryBarrier = {};

			imageMemoryBarrier.oldLayout = vk::ImageLayout::eGeneral;
			imageMemoryBarrier.newLayout = vk::ImageLayout::eGeneral;
			imageMemoryBarrier.image = textureComputeTarget.image;
			imageMemoryBarrier.subresourceRange = vk::ImageSubresourceRange{ vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 };
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

			vk::Viewport viewport = vks::initializers::viewport((float)width, (float)height, 0.0f, 1.0f);
			drawCmdBuffers[i].setViewport(0, viewport);

			vk::Rect2D scissor = vks::initializers::rect2D(width, height, 0, 0);
			drawCmdBuffers[i].setScissor(0, scissor);

			// Display ray traced image generated by compute shader as a full screen quad
			// Quad vertices are generated in the vertex shader
			drawCmdBuffers[i].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, graphics.pipelineLayout, 0, graphics.descriptorSet, nullptr);
			drawCmdBuffers[i].bindPipeline(vk::PipelineBindPoint::eGraphics, graphics.pipeline);
			vkCmdDraw(drawCmdBuffers[i], 3, 1, 0, 0);

			drawCmdBuffers[i].endRenderPass();

			drawCmdBuffers[i].end();
		}

	}

	void buildComputeCommandBuffer()
	{
		vk::CommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();

		compute.commandBuffer.begin(cmdBufInfo);

		compute.commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, compute.pipeline);
		compute.commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, compute.pipelineLayout, 0, compute.descriptorSet, nullptr);

		vkCmdDispatch(compute.commandBuffer, textureComputeTarget.width / 16, textureComputeTarget.height / 16, 1);

		vkEndCommandBuffer(compute.commandBuffer);
	}

	uint32_t currentId = 0;	// Id used to identify objects by the ray tracing shader

	Sphere newSphere(glm::vec3 pos, float radius, glm::vec3 diffuse, float specular)
	{
		Sphere sphere;
		sphere.id = currentId++;
		sphere.pos = pos;
		sphere.radius = radius;
		sphere.diffuse = diffuse;
		sphere.specular = specular;
		return sphere;
	}

	Plane newPlane(glm::vec3 normal, float distance, glm::vec3 diffuse, float specular)
	{
		Plane plane;
		plane.id = currentId++;
		plane.normal = normal;
		plane.distance = distance;
		plane.diffuse = diffuse;
		plane.specular = specular;
		return plane;
	}

	// Setup and fill the compute shader storage buffers containing primitives for the raytraced scene
	void prepareStorageBuffers()
	{
		// Spheres
		std::vector<Sphere> spheres;
		spheres.push_back(newSphere(glm::vec3(1.75f, -0.5f, 0.0f), 1.0f, glm::vec3(0.0f, 1.0f, 0.0f), 32.0f));
		spheres.push_back(newSphere(glm::vec3(0.0f, 1.0f, -0.5f), 1.0f, glm::vec3(0.65f, 0.77f, 0.97f), 32.0f));
		spheres.push_back(newSphere(glm::vec3(-1.75f, -0.75f, -0.5f), 1.25f, glm::vec3(0.9f, 0.76f, 0.46f), 32.0f));
		vk::DeviceSize storageBufferSize = spheres.size() * sizeof(Sphere);

		// Stage
		vks::Buffer stagingBuffer;

		vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eTransferSrc,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			&stagingBuffer,
			storageBufferSize,
			spheres.data());

		vulkanDevice->createBuffer(
			// The SSBO will be used as a storage buffer for the compute pipeline and as a vertex buffer in the graphics pipeline
			vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
			vk::MemoryPropertyFlagBits::eDeviceLocal,
			&compute.storageBuffers.spheres,
			storageBufferSize);

		// Copy to staging buffer
		vk::CommandBuffer copyCmd = VulkanExampleBase::createCommandBuffer(vk::CommandBufferLevel::ePrimary, true);
		vk::BufferCopy copyRegion = {};
		copyRegion.size = storageBufferSize;
		copyCmd.copyBuffer(stagingBuffer.buffer, compute.storageBuffers.spheres.buffer, copyRegion);
		VulkanExampleBase::flushCommandBuffer(copyCmd, queue, true);

		stagingBuffer.destroy();

		// Planes
		std::vector<Plane> planes;
		const float roomDim = 4.0f;
		planes.push_back(newPlane(glm::vec3(0.0f, 1.0f, 0.0f), roomDim, glm::vec3(1.0f), 32.0f));
		planes.push_back(newPlane(glm::vec3(0.0f, -1.0f, 0.0f), roomDim, glm::vec3(1.0f), 32.0f));
		planes.push_back(newPlane(glm::vec3(0.0f, 0.0f, 1.0f), roomDim, glm::vec3(1.0f), 32.0f));
		planes.push_back(newPlane(glm::vec3(0.0f, 0.0f, -1.0f), roomDim, glm::vec3(0.0f), 32.0f));
		planes.push_back(newPlane(glm::vec3(-1.0f, 0.0f, 0.0f), roomDim, glm::vec3(1.0f, 0.0f, 0.0f), 32.0f));
		planes.push_back(newPlane(glm::vec3(1.0f, 0.0f, 0.0f), roomDim, glm::vec3(0.0f, 1.0f, 0.0f), 32.0f));
		storageBufferSize = planes.size() * sizeof(Plane);

		// Stage
		vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eTransferSrc,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			&stagingBuffer,
			storageBufferSize,
			planes.data());

		vulkanDevice->createBuffer(
			// The SSBO will be used as a storage buffer for the compute pipeline and as a vertex buffer in the graphics pipeline
			vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
			vk::MemoryPropertyFlagBits::eDeviceLocal,
			&compute.storageBuffers.planes,
			storageBufferSize);

		// Copy to staging buffer
		copyCmd = VulkanExampleBase::createCommandBuffer(vk::CommandBufferLevel::ePrimary, true);
		copyRegion.size = storageBufferSize;
		copyCmd.copyBuffer(stagingBuffer.buffer, compute.storageBuffers.planes.buffer, copyRegion);
		VulkanExampleBase::flushCommandBuffer(copyCmd, queue, true);

		stagingBuffer.destroy();
	}

	void setupDescriptorPool()
	{
		std::vector<vk::DescriptorPoolSize> poolSizes =
		{
			vks::initializers::descriptorPoolSize(vk::DescriptorType::eUniformBuffer, 2),			// Compute UBO
			vks::initializers::descriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, 4),	// Graphics image samplers
			vks::initializers::descriptorPoolSize(vk::DescriptorType::eStorageImage, 1),				// Storage image for ray traced image output
			vks::initializers::descriptorPoolSize(vk::DescriptorType::eStorageBuffer, 2),			// Storage buffer for the scene primitives
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
			// Binding 0 : Fragment shader image sampler
			vks::initializers::descriptorSetLayoutBinding(
				vk::DescriptorType::eCombinedImageSampler,
				vk::ShaderStageFlagBits::eFragment,
				0)
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

		graphics.descriptorSet = device.allocateDescriptorSets(allocInfo)[0];

		std::vector<vk::WriteDescriptorSet> writeDescriptorSets =
		{
			// Binding 0 : Fragment shader texture sampler
			vks::initializers::writeDescriptorSet(
				graphics.descriptorSet,
				vk::DescriptorType::eCombinedImageSampler,
				0,
				&textureComputeTarget.descriptor)
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
				vk::CullModeFlagBits::eFront,
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

		// Display pipeline
		std::array<vk::PipelineShaderStageCreateInfo,2> shaderStages;

		shaderStages[0] = loadShader(getAssetPath() + "shaders/raytracing/texture.vert.spv", vk::ShaderStageFlagBits::eVertex);
		shaderStages[1] = loadShader(getAssetPath() + "shaders/raytracing/texture.frag.spv", vk::ShaderStageFlagBits::eFragment);

		vk::GraphicsPipelineCreateInfo pipelineCreateInfo =
			vks::initializers::pipelineCreateInfo(
				graphics.pipelineLayout,
				renderPass);

		vk::PipelineVertexInputStateCreateInfo emptyInputState{};

		emptyInputState.vertexAttributeDescriptionCount = 0;
		emptyInputState.pVertexAttributeDescriptions = nullptr;
		emptyInputState.vertexBindingDescriptionCount = 0;
		emptyInputState.pVertexBindingDescriptions = nullptr;
		pipelineCreateInfo.pVertexInputState = &emptyInputState;

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

	// Prepare the compute pipeline that generates the ray traced image
	void prepareCompute()
	{
		// Create a compute capable device queue
		// The VulkanDevice::createLogicalDevice functions finds a compute capable queue and prefers queue families that only support compute
		// Depending on the implementation this may result in different queue family indices for graphics and computes,
		// requiring proper synchronization (see the memory barriers in buildComputeCommandBuffer)
		/*vk::DeviceQueueCreateInfo queueCreateInfo = {};
		queueCreateInfo.queueFamilyIndex = vulkanDevice->queueFamilyIndices.compute;
		queueCreateInfo.queueCount = 1;
		*/
		compute.queue = device.getQueue(vulkanDevice->queueFamilyIndices.compute, 0);

		std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings = {
			// Binding 0: Storage image (raytraced output)
			vks::initializers::descriptorSetLayoutBinding(
				vk::DescriptorType::eStorageImage,
				vk::ShaderStageFlagBits::eCompute,
				0),
			// Binding 1: Uniform buffer block
			vks::initializers::descriptorSetLayoutBinding(
				vk::DescriptorType::eUniformBuffer,
				vk::ShaderStageFlagBits::eCompute,
				1),
			// Binding 1: Shader storage buffer for the spheres
			vks::initializers::descriptorSetLayoutBinding(
				vk::DescriptorType::eStorageBuffer,
				vk::ShaderStageFlagBits::eCompute,
				2),
			// Binding 1: Shader storage buffer for the planes
			vks::initializers::descriptorSetLayoutBinding(
				vk::DescriptorType::eStorageBuffer,
				vk::ShaderStageFlagBits::eCompute,
				3)
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
			// Binding 0: Output storage image
			vks::initializers::writeDescriptorSet(
				compute.descriptorSet,
				vk::DescriptorType::eStorageImage,
				0,
				&textureComputeTarget.descriptor),
			// Binding 1: Uniform buffer block
			vks::initializers::writeDescriptorSet(
				compute.descriptorSet,
				vk::DescriptorType::eUniformBuffer,
				1,
				&compute.uniformBuffer.descriptor),
			// Binding 2: Shader storage buffer for the spheres
			vks::initializers::writeDescriptorSet(
				compute.descriptorSet,
				vk::DescriptorType::eStorageBuffer,
				2,
				&compute.storageBuffers.spheres.descriptor),
			// Binding 2: Shader storage buffer for the planes
			vks::initializers::writeDescriptorSet(
				compute.descriptorSet,
				vk::DescriptorType::eStorageBuffer,
				3,
				&compute.storageBuffers.planes.descriptor)
		};

		device.updateDescriptorSets(computeWriteDescriptorSets, nullptr);

		// Create compute shader pipelines
		vk::ComputePipelineCreateInfo computePipelineCreateInfo =
			vks::initializers::computePipelineCreateInfo(
				compute.pipelineLayout);

		computePipelineCreateInfo.stage = loadShader(getAssetPath() + "shaders/raytracing/raytracing.comp.spv", vk::ShaderStageFlagBits::eCompute);
		compute.pipeline = device.createComputePipelines(pipelineCache, computePipelineCreateInfo)[0];

		// Separate command pool as queue family for compute may be different than graphics
		vk::CommandPoolCreateInfo cmdPoolInfo = {};

		cmdPoolInfo.queueFamilyIndex = vulkanDevice->queueFamilyIndices.compute;
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
		// Compute shader parameter uniform buffer block
		vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eUniformBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			&compute.uniformBuffer,
			sizeof(compute.ubo));

		updateUniformBuffers();
	}

	void updateUniformBuffers()
	{
		compute.ubo.lightPos.x = 0.0f + sin(glm::radians(timer * 360.0f)) * cos(glm::radians(timer * 360.0f)) * 2.0f;
		compute.ubo.lightPos.y = 0.0f + sin(glm::radians(timer * 360.0f)) * 2.0f;
		compute.ubo.lightPos.z = 0.0f + cos(glm::radians(timer * 360.0f)) * 2.0f;
		compute.ubo.camera.pos = camera.position * -1.0f;
		compute.uniformBuffer.map();
		memcpy(compute.uniformBuffer.mapped, &compute.ubo, sizeof(compute.ubo));
		compute.uniformBuffer.unmap();
	}

	void draw()
	{
		VulkanExampleBase::prepareFrame();

		// Command buffer to be sumitted to the queue
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &drawCmdBuffers[currentBuffer];
		queue.submit(submitInfo, vk::Fence(nullptr));

		VulkanExampleBase::submitFrame();

		// Submit compute commands
		// Use a fence to ensure that compute command buffer has finished executing before using it again
		device.waitForFences(compute.fence, VK_TRUE, UINT64_MAX);
		device.resetFences(compute.fence);

		vk::SubmitInfo computeSubmitInfo = vks::initializers::submitInfo();
		computeSubmitInfo.commandBufferCount = 1;
		computeSubmitInfo.pCommandBuffers = &compute.commandBuffer;

		compute.queue.submit(computeSubmitInfo, compute.fence);
	}

	void prepare()
	{
		VulkanExampleBase::prepare();
		prepareStorageBuffers();
		prepareUniformBuffers();
		prepareTextureTarget(&textureComputeTarget, TEX_DIM, TEX_DIM, vk::Format::eR8G8B8A8Unorm);
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
		if (!paused)
		{
			updateUniformBuffers();
		}
	}

	virtual void viewChanged()
	{
		compute.ubo.aspectRatio = (float)width / (float)height;
		updateUniformBuffers();
	}
};

VULKAN_EXAMPLE_MAIN()