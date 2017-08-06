/*
* Vulkan Example - Compute shader N-body simulation using two passes and shared compute shader memory
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
#include <random>

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
// Lower particle count on Android for performance reasons
#define PARTICLES_PER_ATTRACTOR 3 * 1024
#else
#define PARTICLES_PER_ATTRACTOR 4 * 1024
#endif

class VulkanExample : public VulkanExampleBase
{
public:
	uint32_t numParticles;

	struct {
		vks::Texture2D particle;
		vks::Texture2D gradient;
	} textures;

	struct {
		vk::PipelineVertexInputStateCreateInfo inputState;
		std::vector<vk::VertexInputBindingDescription> bindingDescriptions;
		std::vector<vk::VertexInputAttributeDescription> attributeDescriptions;
	} vertices;

	// Resources for the graphics part of the example
	struct {
		vks::Buffer uniformBuffer;					// Contains scene matrices
		vk::DescriptorSetLayout descriptorSetLayout;	// Particle system rendering shader binding layout
		vk::DescriptorSet descriptorSet;				// Particle system rendering shader bindings
		vk::PipelineLayout pipelineLayout;			// Layout of the graphics pipeline
		vk::Pipeline pipeline;						// Particle rendering pipeline
		struct {
			glm::mat4 projection;
			glm::mat4 view;
			glm::vec2 screenDim;
		} ubo;
	} graphics;

	// Resources for the compute part of the example
	struct {
		vks::Buffer storageBuffer;					// (Shader) storage buffer object containing the particles
		vks::Buffer uniformBuffer;					// Uniform buffer object containing particle system parameters
		vk::Queue queue;								// Separate queue for compute commands (queue family may differ from the one used for graphics)
		vk::CommandPool commandPool;					// Use a separate command pool (queue family may differ from the one used for graphics)
		vk::CommandBuffer commandBuffer;				// Command buffer storing the dispatch commands and barriers
		vk::Fence fence;								// Synchronization fence to avoid rewriting compute CB if still in use
		vk::DescriptorSetLayout descriptorSetLayout;	// Compute shader binding layout
		vk::DescriptorSet descriptorSet;				// Compute shader bindings
		vk::PipelineLayout pipelineLayout;			// Layout of the compute pipeline
		vk::Pipeline pipelineCalculate;				// Compute pipeline for N-Body velocity calculation (1st pass)
		vk::Pipeline pipelineIntegrate;				// Compute pipeline for euler integration (2nd pass)
		vk::Pipeline blur;
		vk::PipelineLayout pipelineLayoutBlur;
		vk::DescriptorSetLayout descriptorSetLayoutBlur;
		vk::DescriptorSet descriptorSetBlur;
		struct computeUBO {							// Compute shader uniform block object
			float deltaT;							//		Frame delta time
			float destX;							//		x position of the attractor
			float destY;							//		y position of the attractor
			int32_t particleCount;
		} ubo;
	} compute;

	// SSBO particle declaration
	struct Particle {
		glm::vec4 pos;								// xyz = position, w = mass
		glm::vec4 vel;								// xyz = velocity, w = gradient texture position
	};

	VulkanExample() : VulkanExampleBase(ENABLE_VALIDATION)
	{
		enableTextOverlay = true;
		title = "Vulkan Example - Compute shader N-body system";

		camera.type = Camera::CameraType::lookat;
		camera.setPerspective(60.0f, (float)width / (float)height, 0.1f, 512.0f);
		camera.setRotation(glm::vec3(-26.0f, 75.0f, 0.0f));
		camera.setTranslation(glm::vec3(0.0f, 0.0f, -14.0f));
		camera.movementSpeed = 2.5f;
	}

	~VulkanExample()
	{
		// Graphics
		graphics.uniformBuffer.destroy();
		device.destroyPipeline(graphics.pipeline);
		device.destroyPipelineLayout(graphics.pipelineLayout);
		device.destroyDescriptorSetLayout(graphics.descriptorSetLayout);

		// Compute
		compute.storageBuffer.destroy();
		compute.uniformBuffer.destroy();
		device.destroyPipelineLayout(compute.pipelineLayout);
		device.destroyDescriptorSetLayout(compute.descriptorSetLayout);
		device.destroyPipeline(compute.pipelineCalculate);
		device.destroyPipeline(compute.pipelineIntegrate);
		device.destroyFence(compute.fence);
		device.destroyCommandPool(compute.commandPool);

		textures.particle.destroy();
		textures.gradient.destroy();
	}

	void loadAssets()
	{
		textures.particle.loadFromFile(getAssetPath() + "textures/particle01_rgba.ktx", vk::Format::eR8G8B8A8Unorm, vulkanDevice, queue);
		textures.gradient.loadFromFile(getAssetPath() + "textures/particle_gradient_rgba.ktx", vk::Format::eR8G8B8A8Unorm, vulkanDevice, queue);
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
		clearValues[0].color = { {0.0f, 0.0f, 0.0f, 1.0f} };
		clearValues[1].depthStencil = { 1.0f, 0 };

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

			// Draw the particle system using the update vertex buffer

			drawCmdBuffers[i].beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);

			vk::Viewport viewport = vks::initializers::viewport((float)width, (float)height, 0.0f, 1.0f);
			drawCmdBuffers[i].setViewport(0, viewport);

			vk::Rect2D scissor = vks::initializers::rect2D(width, height, 0, 0);
			vkCmdSetScissor(drawCmdBuffers[i], 0, 1, &scissor);

			drawCmdBuffers[i].bindPipeline(vk::PipelineBindPoint::eGraphics, graphics.pipeline);
			drawCmdBuffers[i].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, graphics.pipelineLayout, 0, graphics.descriptorSet, nullptr);

			vk::DeviceSize offsets[1] = { 0 };
			drawCmdBuffers[i].bindVertexBuffers(VERTEX_BUFFER_BIND_ID, compute.storageBuffer.buffer, offsets);
			vkCmdDraw(drawCmdBuffers[i], numParticles, 1, 0, 0);

			drawCmdBuffers[i].endRenderPass();

			drawCmdBuffers[i].end();
		}

	}

	void buildComputeCommandBuffer()
	{
		vk::CommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();

		compute.commandBuffer.begin(cmdBufInfo);

		// Compute particle movement

		// Add memory barrier to ensure that the (graphics) vertex shader has fetched attributes before compute starts to write to the buffer
		vk::BufferMemoryBarrier bufferBarrier = vks::initializers::bufferMemoryBarrier();
		bufferBarrier.buffer = compute.storageBuffer.buffer;
		bufferBarrier.size = compute.storageBuffer.descriptor.range;
		bufferBarrier.srcAccessMask = vk::AccessFlagBits::eVertexAttributeRead;						// Vertex shader invocations have finished reading from the buffer
		bufferBarrier.dstAccessMask = vk::AccessFlagBits::eShaderWrite;								// Compute shader wants to write to the buffer
		// Transfer ownership if compute and graphics queue familiy indices differ
		bufferBarrier.srcQueueFamilyIndex = vulkanDevice->queueFamilyIndices.graphics;			
		bufferBarrier.dstQueueFamilyIndex = vulkanDevice->queueFamilyIndices.compute;			

		vkCmdPipelineBarrier(
			compute.commandBuffer,
			vk::PipelineStageFlagBits::eVertexShader,
			vk::PipelineStageFlagBits::eComputeShader,
			VK_FLAGS_NONE,
			0, nullptr,
			1, &bufferBarrier,
			0, nullptr);

		compute.commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, compute.pipelineCalculate);
		compute.commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, compute.pipelineLayout, 0, compute.descriptorSet, nullptr);

		// First pass: Calculate particle movement
		// -------------------------------------------------------------------------------------------------------
		vkCmdDispatch(compute.commandBuffer, numParticles / 256, 1, 1);

		// Add memory barrier to ensure that compute shader has finished writing to the buffer 
		bufferBarrier.srcAccessMask = vk::AccessFlagBits::eShaderWrite;								// Compute shader has finished writes to the buffer
		bufferBarrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;					
		bufferBarrier.buffer = compute.storageBuffer.buffer;
		bufferBarrier.size = compute.storageBuffer.descriptor.range;
		// No ownership transfer necessary
		bufferBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED; 
		bufferBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

		vkCmdPipelineBarrier(
			compute.commandBuffer,
			vk::PipelineStageFlagBits::eComputeShader,
			vk::PipelineStageFlagBits::eComputeShader,
			VK_FLAGS_NONE,
			0, nullptr,
			1, &bufferBarrier,
			0, nullptr);

		// Second pass: Integrate particles
		// -------------------------------------------------------------------------------------------------------
		compute.commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, compute.pipelineIntegrate);
		vkCmdDispatch(compute.commandBuffer, numParticles / 256, 1, 1);

		// Add memory barrier to ensure that compute shader has finished writing to the buffer
		// Without this the (rendering) vertex shader may display incomplete results (partial data from last frame) 
		bufferBarrier.srcAccessMask = vk::AccessFlagBits::eShaderWrite;								// Compute shader has finished writes to the buffer
		bufferBarrier.dstAccessMask = vk::AccessFlagBits::eVertexAttributeRead;						// Vertex shader invocations want to read from the buffer
		bufferBarrier.buffer = compute.storageBuffer.buffer;
		bufferBarrier.size = compute.storageBuffer.descriptor.range;
		// Transfer ownership if compute and graphics queue familiy indices differ
		bufferBarrier.srcQueueFamilyIndex = vulkanDevice->queueFamilyIndices.compute;
		bufferBarrier.dstQueueFamilyIndex = vulkanDevice->queueFamilyIndices.graphics;

		vkCmdPipelineBarrier(
			compute.commandBuffer,
			vk::PipelineStageFlagBits::eComputeShader,
			vk::PipelineStageFlagBits::eVertexShader,
			VK_FLAGS_NONE,
			0, nullptr,
			1, &bufferBarrier,
			0, nullptr);

		vkEndCommandBuffer(compute.commandBuffer);
	}

	// Setup and fill the compute shader storage buffers containing the particles
	void prepareStorageBuffers()
	{
#if 0
		std::vector<glm::vec3> attractors = {
			glm::vec3(2.5f, 1.5f, 0.0f),
			glm::vec3(-2.5f, -1.5f, 0.0f),
		};
#else
		std::vector<glm::vec3> attractors = {
			glm::vec3(5.0f, 0.0f, 0.0f),
			glm::vec3(-5.0f, 0.0f, 0.0f),
			glm::vec3(0.0f, 0.0f, 5.0f),
			glm::vec3(0.0f, 0.0f, -5.0f),
			glm::vec3(0.0f, 4.0f, 0.0f),
			glm::vec3(0.0f, -8.0f, 0.0f),
		};
#endif

		numParticles = static_cast<uint32_t>(attractors.size()) * PARTICLES_PER_ATTRACTOR;

		// Initial particle positions
		std::vector<Particle> particleBuffer(numParticles);

		std::mt19937 rndGen(static_cast<uint32_t>(time(0)));
		std::normal_distribution<float> rndDist(0.0f, 1.0f);

		for (uint32_t i = 0; i < static_cast<uint32_t>(attractors.size()); i++)
		{
			for (uint32_t j = 0; j < PARTICLES_PER_ATTRACTOR; j++)
			{
				Particle &particle = particleBuffer[i * PARTICLES_PER_ATTRACTOR + j];

				// First particle in group as heavy center of gravity
				if (j == 0)
				{
					particle.pos = glm::vec4(attractors[i] * 1.5f, 90000.0f);
					particle.vel = glm::vec4(glm::vec4(0.0f));
				}
				else
				{
					// Position					
					glm::vec3 position(attractors[i] + glm::vec3(rndDist(rndGen), rndDist(rndGen), rndDist(rndGen)) * 0.75f);
					float len = glm::length(glm::normalize(position - attractors[i]));
					position.y *= 2.0f - (len * len);

					// Velocity
					glm::vec3 angular = glm::vec3(0.5f, 1.5f, 0.5f) * (((i % 2) == 0) ? 1.0f : -1.0f);
					glm::vec3 velocity = glm::cross((position - attractors[i]), angular) + glm::vec3(rndDist(rndGen), rndDist(rndGen), rndDist(rndGen) * 0.025f);

					float mass = (rndDist(rndGen) * 0.5f + 0.5f) * 75.0f;
					particle.pos = glm::vec4(position, mass);
					particle.vel = glm::vec4(velocity, 0.0f);
				}

				// Color gradient offset
				particle.vel.w = (float)i * 1.0f / static_cast<uint32_t>(attractors.size());
			}
		}

		compute.ubo.particleCount = numParticles;

		vk::DeviceSize storageBufferSize = particleBuffer.size() * sizeof(Particle);

		// Staging
		// SSBO won't be changed on the host after upload so copy to device local memory 

		vks::Buffer stagingBuffer;

		vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eTransferSrc,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			&stagingBuffer,
			storageBufferSize,
			particleBuffer.data());

		vulkanDevice->createBuffer(
			// The SSBO will be used as a storage buffer for the compute pipeline and as a vertex buffer in the graphics pipeline
			vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
			vk::MemoryPropertyFlagBits::eDeviceLocal,
			&compute.storageBuffer,
			storageBufferSize);

		// Copy to staging buffer
		vk::CommandBuffer copyCmd = VulkanExampleBase::createCommandBuffer(vk::CommandBufferLevel::ePrimary, true);
		vk::BufferCopy copyRegion = {};
		copyRegion.size = storageBufferSize;
		copyCmd.copyBuffer(stagingBuffer.buffer, compute.storageBuffer.buffer, copyRegion);
		VulkanExampleBase::flushCommandBuffer(copyCmd, queue, true);

		stagingBuffer.destroy();

		// Binding description
		vertices.bindingDescriptions.resize(1);
		vertices.bindingDescriptions[0] =
			vks::initializers::vertexInputBindingDescription(
				VERTEX_BUFFER_BIND_ID,
				sizeof(Particle),
				vk::VertexInputRate::eVertex);

		// Attribute descriptions
		// Describes memory layout and shader positions
		vertices.attributeDescriptions.resize(2);
		// Location 0 : Position
		vertices.attributeDescriptions[0] =
			vks::initializers::vertexInputAttributeDescription(
				VERTEX_BUFFER_BIND_ID,
				0,
				vk::Format::eR32G32B32A32Sfloat,
				offsetof(Particle, pos));
		// Location 1 : Velocity (used for gradient lookup)
		vertices.attributeDescriptions[1] =
			vks::initializers::vertexInputAttributeDescription(
				VERTEX_BUFFER_BIND_ID,
				1,
				vk::Format::eR32G32B32A32Sfloat,
				offsetof(Particle, vel));

		// Assign to vertex buffer
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
			vks::initializers::descriptorPoolSize(vk::DescriptorType::eUniformBuffer, 2),
			vks::initializers::descriptorPoolSize(vk::DescriptorType::eStorageBuffer, 1),
			vks::initializers::descriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, 2)
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
		std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings;
		setLayoutBindings = {
			vks::initializers::descriptorSetLayoutBinding(vk::DescriptorType::eCombinedImageSampler, vk::ShaderStageFlagBits::eFragment, 0),
			vks::initializers::descriptorSetLayoutBinding(vk::DescriptorType::eCombinedImageSampler, vk::ShaderStageFlagBits::eFragment, 1),
			vks::initializers::descriptorSetLayoutBinding(vk::DescriptorType::eUniformBuffer, vk::ShaderStageFlagBits::eVertex, 2),
		};

		vk::DescriptorSetLayoutCreateInfo descriptorLayout =
			vks::initializers::descriptorSetLayoutCreateInfo(
				setLayoutBindings.data(),
				static_cast<uint32_t>(setLayoutBindings.size()));

		graphics.descriptorSetLayout = device.createDescriptorSetLayout(descriptorLayout);

		vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo =
			vks::initializers::pipelineLayoutCreateInfo(
				&graphics.descriptorSetLayout,
				1);

		graphics.pipelineLayout = device.createPipelineLayout(pipelineLayoutCreateInfo);
	}

	void setupDescriptorSet()
	{
		vk::DescriptorSetAllocateInfo allocInfo =
			vks::initializers::descriptorSetAllocateInfo(
				descriptorPool,
				&graphics.descriptorSetLayout,
				1);

		graphics.descriptorSet = device.allocateDescriptorSets(allocInfo)[0];

		std::vector<vk::WriteDescriptorSet> writeDescriptorSets;
		writeDescriptorSets = {
			vks::initializers::writeDescriptorSet(graphics.descriptorSet, vk::DescriptorType::eCombinedImageSampler, 0, &textures.particle.descriptor),
			vks::initializers::writeDescriptorSet(graphics.descriptorSet, vk::DescriptorType::eCombinedImageSampler, 1, &textures.gradient.descriptor),
			vks::initializers::writeDescriptorSet(graphics.descriptorSet, vk::DescriptorType::eUniformBuffer, 2, &graphics.uniformBuffer.descriptor),
		};
		device.updateDescriptorSets(writeDescriptorSets);
	}

	void preparePipelines()
	{
		vk::PipelineInputAssemblyStateCreateInfo inputAssemblyState =
			vks::initializers::pipelineInputAssemblyStateCreateInfo(
				vk::PrimitiveTopology::ePointList,
				0,
				VK_FALSE);

		vk::PipelineRasterizationStateCreateInfo rasterizationState =
			vks::initializers::pipelineRasterizationStateCreateInfo(
				vk::PolygonMode::eFill,
				vk::CullModeFlagBits::eNone,
				vk::FrontFace::eCounterClockwise,
				0);

		vk::PipelineColorBlendAttachmentState blendAttachmentState =
			vks::initializers::pipelineColorBlendAttachmentState(
				0xf,
				VK_FALSE);

		vk::PipelineColorBlendStateCreateInfo colorBlendState =
			vks::initializers::pipelineColorBlendStateCreateInfo(
				1,
				&blendAttachmentState);

		vk::PipelineDepthStencilStateCreateInfo depthStencilState =
			vks::initializers::pipelineDepthStencilStateCreateInfo(
				VK_FALSE,
				VK_FALSE,
				vk::CompareOp::eAlways);

		vk::PipelineViewportStateCreateInfo viewportState =
			vks::initializers::pipelineViewportStateCreateInfo(1, 1, 0);

		vk::PipelineMultisampleStateCreateInfo multisampleState =
			vks::initializers::pipelineMultisampleStateCreateInfo(
				vk::SampleCountFlagBits::e1,
				0);

		std::vector<vk::DynamicState> dynamicStateEnables = {
			vk::DynamicState::eViewport,
			vk::DynamicState::eScissor
		};
		vk::PipelineDynamicStateCreateInfo dynamicState =
			vks::initializers::pipelineDynamicStateCreateInfo(
				dynamicStateEnables.data(),
				static_cast<uint32_t>(dynamicStateEnables.size()),
				0);

		// Rendering pipeline
		// Load shaders
		std::array<vk::PipelineShaderStageCreateInfo,2> shaderStages;

		shaderStages[0] = loadShader(getAssetPath() + "shaders/computenbody/particle.vert.spv", vk::ShaderStageFlagBits::eVertex);
		shaderStages[1] = loadShader(getAssetPath() + "shaders/computenbody/particle.frag.spv", vk::ShaderStageFlagBits::eFragment);

		vk::GraphicsPipelineCreateInfo pipelineCreateInfo =
			vks::initializers::pipelineCreateInfo(
				graphics.pipelineLayout,
				renderPass,
				0);

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
		pipelineCreateInfo.renderPass = renderPass;

		// Additive blending
		blendAttachmentState.colorWriteMask = 0xF;
		blendAttachmentState.blendEnable = VK_TRUE;
		blendAttachmentState.colorBlendOp = vk::BlendOp::eAdd;
		blendAttachmentState.srcColorBlendFactor = vk::BlendFactor::eOne;
		blendAttachmentState.dstColorBlendFactor = vk::BlendFactor::eOne;
		blendAttachmentState.alphaBlendOp = vk::BlendOp::eAdd;
		blendAttachmentState.srcAlphaBlendFactor = vk::BlendFactor::eSrcAlpha;
		blendAttachmentState.dstAlphaBlendFactor = vk::BlendFactor::eDstAlpha;

		graphics.pipeline = device.createGraphicsPipelines(pipelineCache, pipelineCreateInfo)[0];
	}

	void prepareCompute()
	{
		// Create a compute capable device queue
		// The VulkanDevice::createLogicalDevice functions finds a compute capable queue and prefers queue families that only support compute
		// Depending on the implementation this may result in different queue family indices for graphics and computes,
		// requiring proper synchronization (see the memory barriers in buildComputeCommandBuffer)
		//vk::DeviceQueueCreateInfo queueCreateInfo = {};

		//queueCreateInfo.pNext = NULL;
		//queueCreateInfo.queueFamilyIndex = vulkanDevice->queueFamilyIndices.compute;
		//queueCreateInfo.queueCount = 1;
		compute.queue = device.getQueue(vulkanDevice->queueFamilyIndices.compute);

		// Create compute pipeline
		// Compute pipelines are created separate from graphics pipelines even if they use the same queue (family index)

		std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings = {
			// Binding 0 : Particle position storage buffer
			vks::initializers::descriptorSetLayoutBinding(
				vk::DescriptorType::eStorageBuffer,
				vk::ShaderStageFlagBits::eCompute,
				0),
			// Binding 1 : Uniform buffer
			vks::initializers::descriptorSetLayoutBinding(
				vk::DescriptorType::eUniformBuffer,
				vk::ShaderStageFlagBits::eCompute,
				1),
		};

		vk::DescriptorSetLayoutCreateInfo descriptorLayout =
			vks::initializers::descriptorSetLayoutCreateInfo(
				setLayoutBindings.data(),
				static_cast<uint32_t>(setLayoutBindings.size()));

		VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device,	&descriptorLayout, nullptr,	&compute.descriptorSetLayout));

		vk::PipelineLayoutCreateInfo pPipelineLayoutCreateInfo =
			vks::initializers::pipelineLayoutCreateInfo(
				&compute.descriptorSetLayout,
				1);

		VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pPipelineLayoutCreateInfo, nullptr,	&compute.pipelineLayout));

		vk::DescriptorSetAllocateInfo allocInfo =
			vks::initializers::descriptorSetAllocateInfo(
				descriptorPool,
				&compute.descriptorSetLayout,
				1);

		compute.descriptorSet = device.allocateDescriptorSets(allocInfo)[0];

		std::vector<vk::WriteDescriptorSet> computeWriteDescriptorSets =
		{
			// Binding 0 : Particle position storage buffer
			vks::initializers::writeDescriptorSet(
				compute.descriptorSet,
				vk::DescriptorType::eStorageBuffer,
				0,
				&compute.storageBuffer.descriptor),
			// Binding 1 : Uniform buffer
			vks::initializers::writeDescriptorSet(
				compute.descriptorSet,
				vk::DescriptorType::eUniformBuffer,
				1,
				&compute.uniformBuffer.descriptor)
		};

		device.updateDescriptorSets(computeWriteDescriptorSets);

		// Create pipelines
		vk::ComputePipelineCreateInfo computePipelineCreateInfo = vks::initializers::computePipelineCreateInfo(compute.pipelineLayout, 0);

		// 1st pass
		computePipelineCreateInfo.stage = loadShader(getAssetPath() + "shaders/computenbody/particle_calculate.comp.spv", vk::ShaderStageFlagBits::eCompute);

		// Set shader parameters via specialization constants
		struct SpecializationData {
			uint32_t sharedDataSize;
			float gravity;
			float power;
			float soften;
		} specializationData;

		std::vector<vk::SpecializationMapEntry> specializationMapEntries;
		specializationMapEntries.push_back(vks::initializers::specializationMapEntry(0, offsetof(SpecializationData, sharedDataSize), sizeof(uint32_t)));
		specializationMapEntries.push_back(vks::initializers::specializationMapEntry(1, offsetof(SpecializationData, gravity), sizeof(float)));
		specializationMapEntries.push_back(vks::initializers::specializationMapEntry(2, offsetof(SpecializationData, power), sizeof(float)));
		specializationMapEntries.push_back(vks::initializers::specializationMapEntry(3, offsetof(SpecializationData, soften), sizeof(float)));

		specializationData.sharedDataSize = std::min((uint32_t)1024, (uint32_t)(vulkanDevice->properties.limits.maxComputeSharedMemorySize / sizeof(glm::vec4)));

		specializationData.gravity = 0.002f;
		specializationData.power = 0.75f;
		specializationData.soften = 0.05f;

		vk::SpecializationInfo specializationInfo = 
			vks::initializers::specializationInfo(static_cast<uint32_t>(specializationMapEntries.size()), specializationMapEntries.data(), sizeof(specializationData), &specializationData);
		computePipelineCreateInfo.stage.pSpecializationInfo = &specializationInfo;

		compute.pipelineCalculate = device.createComputePipelines(pipelineCache, computePipelineCreateInfo);

		// 2nd pass
		computePipelineCreateInfo.stage = loadShader(getAssetPath() + "shaders/computenbody/particle_integrate.comp.spv", vk::ShaderStageFlagBits::eCompute);
		compute.pipelineIntegrate = device.createComputePipelines(pipelineCache, computePipelineCreateInfo);

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

		compute.commandBuffer) = device.allocateCommandBuffers(cmdBufAllocateInfo);

		// Fence for compute CB sync
		vk::FenceCreateInfo fenceCreateInfo = vks::initializers::fenceCreateInfo(VK_FENCE_CREATE_SIGNALED_BIT);
		compute.fence = device.createFence(fenceCreateInfo);

		// Build a single command buffer containing the compute dispatch commands
		buildComputeCommandBuffer();
	}

	// Prepare and initialize uniform buffer containing shader uniforms
	void prepareUniformBuffers()
	{
		// Compute shader uniform buffer block
		vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eUniformBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			&compute.uniformBuffer,
			sizeof(compute.ubo));

		// Map for host access
		compute.uniformBuffer.map();

		// Vertex shader uniform buffer block
		vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eUniformBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			&graphics.uniformBuffer,
			sizeof(graphics.ubo));

		// Map for host access
		graphics.uniformBuffer.map();

		updateGraphicsUniformBuffers();
	}

	void updateUniformBuffers()
	{
		compute.ubo.deltaT = paused ? 0.0f : frameTimer * 0.05f;
		compute.ubo.destX = sin(glm::radians(timer * 360.0f)) * 0.75f;
		compute.ubo.destY = 0.0f;
		memcpy(compute.uniformBuffer.mapped, &compute.ubo, sizeof(compute.ubo));
	}

	void updateGraphicsUniformBuffers()
	{
		graphics.ubo.projection = camera.matrices.perspective;
		graphics.ubo.view = camera.matrices.view;
		graphics.ubo.screenDim = glm::vec2((float)width, (float)height);
		memcpy(graphics.uniformBuffer.mapped, &graphics.ubo, sizeof(graphics.ubo));
	}

	void draw()
	{
		// Submit graphics commands
		VulkanExampleBase::prepareFrame();

		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &drawCmdBuffers[currentBuffer];
		queue.submit(submitInfo, vk::Fence(nullptr));

		VulkanExampleBase::submitFrame();

		// Submit compute commands
		vkWaitForFences(device, 1, &compute.fence, VK_TRUE, UINT64_MAX);
		device.resetFences(compute.fence);

		vk::SubmitInfo computeSubmitInfo = vks::initializers::submitInfo();
		computeSubmitInfo.commandBufferCount = 1;
		computeSubmitInfo.pCommandBuffers = &compute.commandBuffer;

		compute.queue.submit(computeSubmitInfo, compute.fence);
	}

	void prepare()
	{
		VulkanExampleBase::prepare();
		loadAssets();
		prepareStorageBuffers();
		prepareUniformBuffers();
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
		updateUniformBuffers();
	}

	virtual void viewChanged()
	{
		updateGraphicsUniformBuffers();
	}
};

VULKAN_EXAMPLE_MAIN()