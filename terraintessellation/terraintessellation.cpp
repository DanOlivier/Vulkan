/*
* Vulkan Example - Dynamic terrain tessellation
*
* Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include "VulkanExampleBase.hpp"
#include "VulkanTexture.hpp"
#include "VulkanModel.hpp"
#include "frustum.hpp"

#include <sstream>
#include <iomanip>

#define VERTEX_BUFFER_BIND_ID 0
#define ENABLE_VALIDATION false

class VulkanExample : public VulkanExampleBase
{
public:
	bool wireframe = false;
	bool tessellation = true;

	struct {
		vks::Texture2D heightMap;
		vks::Texture2D skySphere;
		vks::Texture2DArray terrainArray;
	} textures;

	// Vertex layout for the models
	vks::VertexLayout vertexLayout = vks::VertexLayout({
		vks::VERTEX_COMPONENT_POSITION,
		vks::VERTEX_COMPONENT_NORMAL,
		vks::VERTEX_COMPONENT_UV,
	});

	struct {
		vks::Model terrain;
		vks::Model skysphere;
	} models;

	struct {
		vk::PipelineVertexInputStateCreateInfo inputState;
		std::vector<vk::VertexInputBindingDescription> bindingDescriptions;
		std::vector<vk::VertexInputAttributeDescription> attributeDescriptions;
	} vertices;

	struct {
		vks::Buffer terrainTessellation;
		vks::Buffer skysphereVertex;
	} uniformBuffers;

	// Shared values for tessellation control and evaluation stages
	struct {
		glm::mat4 projection;
		glm::mat4 modelview;
		glm::vec4 lightPos = glm::vec4(-48.0f, -40.0f, 46.0f, 0.0f);
		glm::vec4 frustumPlanes[6];
		float displacementFactor = 32.0f;
		float tessellationFactor = 0.75f;
		glm::vec2 viewportDim;
		// Desired size of tessellated quad patch edge
		float tessellatedEdgeSize = 20.0f;
	} uboTess;

	// Skysphere vertex shader stage
	struct {
		glm::mat4 mvp;
	} uboVS;

	struct Pipelines {
		vk::Pipeline terrain;
		vk::Pipeline wireframe;
		vk::Pipeline skysphere;
	} pipelines;

	struct {
		vk::DescriptorSetLayout terrain;
		vk::DescriptorSetLayout skysphere;
	} descriptorSetLayouts;

	struct {
		vk::PipelineLayout terrain;
		vk::PipelineLayout skysphere;
	} pipelineLayouts;

	struct {
		vk::DescriptorSet terrain;
		vk::DescriptorSet skysphere;
	} descriptorSets;

	// Pipeline statistics
	struct {
		vk::Buffer buffer;
		vk::DeviceMemory memory;
	} queryResult;
	vk::QueryPool queryPool;
	
	uint64_t pipelineStats[2] = { 0 };

	// View frustum passed to tessellation control shader for culling
	vks::Frustum frustum;

	VulkanExample() : VulkanExampleBase(ENABLE_VALIDATION)
	{
		enableTextOverlay = true;
		title = "Vulkan Example - Dynamic terrain tessellation";
		camera.type = Camera::CameraType::firstperson;
		camera.setPerspective(60.0f, (float)width / (float)height, 0.1f, 512.0f);
		camera.setRotation(glm::vec3(-12.0f, 159.0f, 0.0f));
		camera.setTranslation(glm::vec3(18.0f, 22.5f, 57.5f));
		camera.movementSpeed = 7.5f;
	}

	~VulkanExample()
	{
		// Clean up used Vulkan resources 
		// Note : Inherited destructor cleans up resources stored in base class
		device.destroyPipeline(pipelines.terrain);
		if (pipelines.wireframe) {
			device.destroyPipeline(pipelines.wireframe);
		}
		device.destroyPipeline(pipelines.skysphere);

		device.destroyPipelineLayout(pipelineLayouts.skysphere);
		device.destroyPipelineLayout(pipelineLayouts.terrain);

		device.destroyDescriptorSetLayout(descriptorSetLayouts.terrain);
		device.destroyDescriptorSetLayout(descriptorSetLayouts.skysphere);

		models.terrain.destroy();
		models.skysphere.destroy();

		uniformBuffers.skysphereVertex.destroy();
		uniformBuffers.terrainTessellation.destroy();

		textures.heightMap.destroy();
		textures.skySphere.destroy();
		textures.terrainArray.destroy();

		if (queryPool) {
			vkDestroyQueryPool(device, queryPool, nullptr);
			device.destroyBuffer(queryResult.buffer);
			device.freeMemory(queryResult.memory);
		}
	}

	// Enable physical device features required for this example				
	virtual void getEnabledFeatures()
	{
		// Tessellation shader support is required for this example
		if (deviceFeatures.tessellationShader) {
			enabledFeatures.tessellationShader = VK_TRUE;
		}
		else {
			vks::tools::exitFatal("Selected GPU does not support tessellation shaders!", "Feature not supported");
		}
		// Fill mode non solid is required for wireframe display
		if (deviceFeatures.fillModeNonSolid) {
			enabledFeatures.fillModeNonSolid = VK_TRUE;
		};
		// Pipeline statistics
		if (deviceFeatures.pipelineStatisticsQuery) {
			enabledFeatures.pipelineStatisticsQuery = VK_TRUE;
		};
	}

	// Setup pool and buffer for storing pipeline statistics results
	void setupQueryResultBuffer()
	{
		uint32_t bufSize = 2 * sizeof(uint64_t);

		vk::MemoryRequirements memReqs;
		vk::MemoryAllocateInfo memAlloc;
		vk::BufferCreateInfo bufferCreateInfo =
			vks::initializers::bufferCreateInfo(
				vk::BufferUsageFlagBits::eUniformBuffer | vk::BufferUsageFlagBits::eTransferDst,
				bufSize);

		// Results are saved in a host visible buffer for easy access by the application
		queryResult.buffer = device.createBuffer(bufferCreateInfo);
		memReqs = device.getBufferMemoryRequirements(queryResult.buffer);
		memAlloc.allocationSize = memReqs.size;
		memAlloc.memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
		queryResult.memory = device.allocateMemory(memAlloc);
		device.bindBufferMemory(queryResult.buffer, queryResult.memory, 0);

		// Create query pool
		if (deviceFeatures.pipelineStatisticsQuery) {
			vk::QueryPoolCreateInfo queryPoolInfo = {};

			queryPoolInfo.queryType = vk::QueryType::ePipelineStatistics;
			queryPoolInfo.pipelineStatistics =
				vk::QueryPipelineStatisticFlagBits::eVertexShaderInvocations |
				vk::QueryPipelineStatisticFlagBits::eTessellationEvaluationShaderInvocations;
			queryPoolInfo.queryCount = 2;
			queryPool = device.createQueryPool(queryPoolInfo);
		}
	}

	// Retrieves the results of the pipeline statistics query submitted to the command buffer
	void getQueryResults()
	{
		// We use vkGetQueryResults to copy the results into a host visible buffer
		device.getQueryPoolResults(
			queryPool,
			0,
			1,
			sizeof(pipelineStats),
			pipelineStats,
			sizeof(uint64_t),
			vk::QueryResultFlagBits::e64);
	}

	void loadAssets()
	{
		models.skysphere.loadFromFile(getAssetPath() + "models/geosphere.obj", vertexLayout, 1.0f, vulkanDevice, queue);

		// Textures
		std::string texFormatSuffix;
		vk::Format texFormat;
		// Get supported compressed texture format
		if (vulkanDevice->features.textureCompressionBC) {
			texFormatSuffix = "_bc3_unorm";
			texFormat = vk::Format::eBc3UnormBlock;
		}
		else if (vulkanDevice->features.textureCompressionASTC_LDR) {
			texFormatSuffix = "_astc_8x8_unorm";
			texFormat = vk::Format::eAstc8x8UnormBlock;
		}
		else if (vulkanDevice->features.textureCompressionETC2) {
			texFormatSuffix = "_etc2_unorm";
			texFormat = vk::Format::eEtc2R8G8B8A8UnormBlock;
		}
		else {
			vks::tools::exitFatal("Device does not support any compressed texture format!", "Error");
		}

		textures.skySphere.loadFromFile(getAssetPath() + "textures/skysphere" + texFormatSuffix + ".ktx", texFormat, vulkanDevice, queue);
		// Terrain textures are stored in a texture array with layers corresponding to terrain height
		textures.terrainArray.loadFromFile(getAssetPath() + "textures/terrain_texturearray" + texFormatSuffix + ".ktx", texFormat, vulkanDevice, queue);

		// Height data is stored in a one-channel texture
		textures.heightMap.loadFromFile(getAssetPath() + "textures/terrain_heightmap_r16.ktx", vk::Format::eR16Unorm, vulkanDevice, queue);

		vk::SamplerCreateInfo samplerInfo = vks::initializers::samplerCreateInfo();

		// Setup a mirroring sampler for the height map
		device.destroySampler(textures.heightMap.sampler);
		samplerInfo.magFilter = vk::Filter::eLinear;
		samplerInfo.minFilter = vk::Filter::eLinear;
		samplerInfo.mipmapMode = vk::SamplerMipmapMode::eLinear;
		samplerInfo.addressModeU = vk::SamplerAddressMode::eMirroredRepeat;
		samplerInfo.addressModeV = samplerInfo.addressModeU;
		samplerInfo.addressModeW = samplerInfo.addressModeU;
		samplerInfo.compareOp = vk::CompareOp::eNever;
		samplerInfo.minLod = 0.0f;
		samplerInfo.maxLod = (float)textures.heightMap.mipLevels;
		samplerInfo.borderColor = vk::BorderColor::eFloatOpaqueWhite;
		textures.heightMap.sampler = device.createSampler(samplerInfo);
		textures.heightMap.descriptor.sampler = textures.heightMap.sampler;

		// Setup a repeating sampler for the terrain texture layers
		device.destroySampler(textures.terrainArray.sampler);
		samplerInfo = vks::initializers::samplerCreateInfo();
		samplerInfo.magFilter = vk::Filter::eLinear;
		samplerInfo.minFilter = vk::Filter::eLinear;
		samplerInfo.mipmapMode = vk::SamplerMipmapMode::eLinear;
		samplerInfo.addressModeU = vk::SamplerAddressMode::eRepeat;
		samplerInfo.addressModeV = samplerInfo.addressModeU;
		samplerInfo.addressModeW = samplerInfo.addressModeU;
		samplerInfo.compareOp = vk::CompareOp::eNever;
		samplerInfo.minLod = 0.0f;
		samplerInfo.maxLod = (float)textures.terrainArray.mipLevels;
		samplerInfo.borderColor = vk::BorderColor::eFloatOpaqueWhite;
		if (deviceFeatures.samplerAnisotropy)
		{
			samplerInfo.maxAnisotropy = 4.0f;
			samplerInfo.anisotropyEnable = VK_TRUE;
		}
		textures.terrainArray.sampler = device.createSampler(samplerInfo);
		textures.terrainArray.descriptor.sampler = textures.terrainArray.sampler;
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
		vk::CommandBufferBeginInfo cmdBufInfo;

		vk::ClearValue clearValues[2];
		clearValues[0].color = defaultClearColor;
		clearValues[0].color = vk::ClearColorValue{ std::array<float, 4>{0.2f, 0.2f, 0.2f, 0.0f} };
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
			renderPassBeginInfo.framebuffer = frameBuffers[i];

			drawCmdBuffers[i].begin(cmdBufInfo);

			if (deviceFeatures.pipelineStatisticsQuery) {
				vkCmdResetQueryPool(drawCmdBuffers[i], queryPool, 0, 2);
			}

			drawCmdBuffers[i].beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);

			vk::Viewport viewport = vks::initializers::viewport((float)width, (float)height, 0.0f, 1.0f);
			drawCmdBuffers[i].setViewport(0, viewport);

			vk::Rect2D scissor = vks::initializers::rect2D(width, height, 0, 0);
			drawCmdBuffers[i].setScissor(0, scissor);

			vkCmdSetLineWidth(drawCmdBuffers[i], 1.0f);

			std::vector<vk::DeviceSize> offsets = { 0 };

			// Skysphere
			drawCmdBuffers[i].bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.skysphere);
			drawCmdBuffers[i].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayouts.skysphere, 0, descriptorSets.skysphere, nullptr);
			drawCmdBuffers[i].bindVertexBuffers(VERTEX_BUFFER_BIND_ID, models.skysphere.vertices.buffer, offsets);
			drawCmdBuffers[i].bindIndexBuffer(models.skysphere.indices.buffer, 0, vk::IndexType::eUint32);
			drawCmdBuffers[i].drawIndexed(models.skysphere.indexCount, 1, 0, 0, 0);

			// Terrrain
			if (deviceFeatures.pipelineStatisticsQuery) {
				// Begin pipeline statistics query		
				vkCmdBeginQuery(drawCmdBuffers[i], queryPool, 0, VK_QUERY_CONTROL_PRECISE_BIT);
			}
			// Render
			drawCmdBuffers[i].bindPipeline(vk::PipelineBindPoint::eGraphics, wireframe ? pipelines.wireframe : pipelines.terrain);
			drawCmdBuffers[i].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayouts.terrain, 0, descriptorSets.terrain, nullptr);
			drawCmdBuffers[i].bindVertexBuffers(VERTEX_BUFFER_BIND_ID, models.terrain.vertices.buffer, offsets);
			drawCmdBuffers[i].bindIndexBuffer(models.terrain.indices.buffer, 0, vk::IndexType::eUint32);
			drawCmdBuffers[i].drawIndexed(models.terrain.indexCount, 1, 0, 0, 0);
			if (deviceFeatures.pipelineStatisticsQuery) {
				// End pipeline statistics query
				vkCmdEndQuery(drawCmdBuffers[i], queryPool, 0);
			}

			drawCmdBuffers[i].endRenderPass();

			drawCmdBuffers[i].end();
		}
	}

	// Encapsulate height map data for easy sampling
	struct HeightMap
	{
	private:
		uint16_t *heightdata;
		uint32_t dim;
		uint32_t scale;
	public:
#if defined(__ANDROID__)
		HeightMap(std::string filename, uint32_t patchsize, AAssetManager* assetManager)
#else
		HeightMap(std::string filename, uint32_t patchsize)
#endif
		{
#if defined(__ANDROID__)
			AAsset* asset = AAssetManager_open(assetManager, filename.c_str(), AASSET_MODE_STREAMING);
			assert(asset);
			size_t size = AAsset_getLength(asset);
			assert(size > 0);
			void *textureData = malloc(size);
			AAsset_read(asset, textureData, size);
			AAsset_close(asset);
			gli::texture2d heightTex(gli::load((const char*)textureData, size));
			free(textureData);
#else
			gli::texture2d heightTex(gli::load(filename));
#endif
			dim = static_cast<uint32_t>(heightTex.extent().x);
			heightdata = new uint16_t[dim * dim];
			memcpy(heightdata, heightTex.data(), heightTex.size());
			this->scale = dim / patchsize;
		};

		~HeightMap()
		{		
			delete[] heightdata;
		}

		float getHeight(uint32_t x, uint32_t y)
		{
			glm::ivec2 rpos = glm::ivec2(x, y) * glm::ivec2(scale);
			rpos.x = std::max(0, std::min(rpos.x, (int)dim-1));
			rpos.y = std::max(0, std::min(rpos.y, (int)dim-1));
			rpos /= glm::ivec2(scale);
			return *(heightdata + (rpos.x + rpos.y * dim) * scale) / 65535.0f;
		}
	};

	// Generate a terrain quad patch for feeding to the tessellation control shader
	void generateTerrain() 
	{
		struct Vertex {
			glm::vec3 pos;
			glm::vec3 normal;
			glm::vec2 uv;
		};

		#define PATCH_SIZE 64
		#define UV_SCALE 1.0f

		Vertex *vertices = new Vertex[PATCH_SIZE * PATCH_SIZE * 4];
			
		const float wx = 2.0f;
		const float wy = 2.0f;

		for (auto x = 0; x < PATCH_SIZE; x++)
		{
			for (auto y = 0; y < PATCH_SIZE; y++)
			{
				uint32_t index = (x + y * PATCH_SIZE);
				vertices[index].pos[0] = x * wx + wx / 2.0f - (float)PATCH_SIZE * wx / 2.0f;
				vertices[index].pos[1] = 0.0f;
				vertices[index].pos[2] = y * wy + wy / 2.0f - (float)PATCH_SIZE * wy / 2.0f;
				vertices[index].uv = glm::vec2((float)x / PATCH_SIZE, (float)y / PATCH_SIZE) * UV_SCALE;
			}
		}

		// Calculate normals from height map using a sobel filter
#if defined(__ANDROID__)
		HeightMap heightMap(getAssetPath() + "textures/terrain_heightmap_r16.ktx", PATCH_SIZE, androidApp->activity->assetManager);
#else
		HeightMap heightMap(getAssetPath() + "textures/terrain_heightmap_r16.ktx", PATCH_SIZE);
#endif
		for (auto x = 0; x < PATCH_SIZE; x++)
		{
			for (auto y = 0; y < PATCH_SIZE; y++)
			{			
				// Get height samples centered around current position
				float heights[3][3];
				for (auto hx = -1; hx <= 1; hx++)
				{
					for (auto hy = -1; hy <= 1; hy++)
					{
						heights[hx+1][hy+1] = heightMap.getHeight(x + hx, y + hy);
					}
				}

				// Calcualte the normal
				glm::vec3 normal;
				// Gx sobel filter
				normal.x = heights[0][0] - heights[2][0] + 2.0f * heights[0][1] - 2.0f * heights[2][1] + heights[0][2] - heights[2][2];
				// Gy sobel filter
				normal.z = heights[0][0] + 2.0f * heights[1][0] + heights[2][0] - heights[0][2] - 2.0f * heights[1][2] - heights[2][2];
				// Calculate missing up component of the normal using the filtered x and y axis
				// The first value controls the bump strength
				normal.y = 0.25f * sqrt( 1.0f - normal.x * normal.x - normal.z * normal.z);

				vertices[x + y * PATCH_SIZE].normal = glm::normalize(normal * glm::vec3(2.0f, 1.0f, 2.0f));
			}
		}

		// Indices
		const uint32_t w = (PATCH_SIZE - 1);
		uint32_t *indices = new uint32_t[w * w * 4];
		for (uint32_t x = 0; x < w; x++)
		{
			for (uint32_t y = 0; y < w; y++)
			{
				uint32_t index = (x + y * w) * 4;
				indices[index] = (x + y * PATCH_SIZE);
				indices[index + 1] = indices[index] + PATCH_SIZE;
				indices[index + 2] = indices[index + 1] + 1;
				indices[index + 3] = indices[index] + 1;
			}
		}
		models.terrain.indexCount = (PATCH_SIZE - 1) * (PATCH_SIZE - 1) * 4;

		uint32_t vertexBufferSize = (PATCH_SIZE * PATCH_SIZE * 4) * sizeof(Vertex);
		uint32_t indexBufferSize = (w * w * 4) * sizeof(uint32_t);

		struct {
			vk::Buffer buffer;
			vk::DeviceMemory memory;
		} vertexStaging, indexStaging;

		// Create staging buffers

		vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eTransferSrc,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			vertexBufferSize,
			&vertexStaging.buffer,
			&vertexStaging.memory,
			vertices);

		vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eTransferSrc,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			indexBufferSize,
			&indexStaging.buffer,
			&indexStaging.memory,
			indices);

		vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst,
			vk::MemoryPropertyFlagBits::eDeviceLocal,
			vertexBufferSize,
			&models.terrain.vertices.buffer,
			&models.terrain.vertices.memory);

		vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst,
			vk::MemoryPropertyFlagBits::eDeviceLocal,
			indexBufferSize,
			&models.terrain.indices.buffer,
			&models.terrain.indices.memory);

		// Copy from staging buffers
		vk::CommandBuffer copyCmd = VulkanExampleBase::createCommandBuffer(vk::CommandBufferLevel::ePrimary, true);

		vk::BufferCopy copyRegion = {};

		copyRegion.size = vertexBufferSize;
		copyCmd.copyBuffer(
			vertexStaging.buffer,
			models.terrain.vertices.buffer,
			copyRegion);

		copyRegion.size = indexBufferSize;
		copyCmd.copyBuffer(
			indexStaging.buffer,
			models.terrain.indices.buffer,
			copyRegion);

		VulkanExampleBase::flushCommandBuffer(copyCmd, queue, true);

		models.terrain.device = device;

		device.destroyBuffer(vertexStaging.buffer);
		device.freeMemory(vertexStaging.memory);
		device.destroyBuffer(indexStaging.buffer);
		device.freeMemory(indexStaging.memory);

		delete[] vertices;
		delete[] indices;
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

		// Location 1 : Normals
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
				sizeof(float) * 6);

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
			vks::initializers::descriptorPoolSize(vk::DescriptorType::eUniformBuffer, 3),
			vks::initializers::descriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, 3)
		};

		vk::DescriptorPoolCreateInfo descriptorPoolInfo =
			vks::initializers::descriptorPoolCreateInfo(
				static_cast<uint32_t>(poolSizes.size()),
				poolSizes.data(),
				2);

		descriptorPool = device.createDescriptorPool(descriptorPoolInfo);
	}

	void setupDescriptorSetLayouts()
	{
		vk::DescriptorSetLayoutCreateInfo descriptorLayout;
		vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo;
		std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings;

		// Terrain
		setLayoutBindings =
		{
			// Binding 0 : Shared Tessellation shader ubo
			vks::initializers::descriptorSetLayoutBinding(
				vk::DescriptorType::eUniformBuffer, 
				vk::ShaderStageFlagBits::eTessellationControl | vk::ShaderStageFlagBits::eTessellationEvaluation,
				0),
			// Binding 1 : Height map
			vks::initializers::descriptorSetLayoutBinding(
				vk::DescriptorType::eCombinedImageSampler,
				vk::ShaderStageFlagBits::eTessellationControl | vk::ShaderStageFlagBits::eTessellationEvaluation | vk::ShaderStageFlagBits::eFragment,
				1),
			// Binding 3 : Terrain texture array layers
			vks::initializers::descriptorSetLayoutBinding(
				vk::DescriptorType::eCombinedImageSampler,
				vk::ShaderStageFlagBits::eFragment,
				2),
		};

		descriptorLayout = vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
		descriptorSetLayouts.terrain = device.createDescriptorSetLayout(descriptorLayout);
		pipelineLayoutCreateInfo = vks::initializers::pipelineLayoutCreateInfo(&descriptorSetLayouts.terrain, 1);
		pipelineLayouts.terrain = device.createPipelineLayout(pipelineLayoutCreateInfo);

		// Skysphere
		setLayoutBindings =
		{
			// Binding 0 : Vertex shader ubo
			vks::initializers::descriptorSetLayoutBinding(
				vk::DescriptorType::eUniformBuffer,
				vk::ShaderStageFlagBits::eVertex,
				0),
			// Binding 1 : Color map
			vks::initializers::descriptorSetLayoutBinding(
				vk::DescriptorType::eCombinedImageSampler,
				vk::ShaderStageFlagBits::eFragment,
				1),
		};

		descriptorLayout = vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
		descriptorSetLayouts.skysphere = device.createDescriptorSetLayout(descriptorLayout);
		pipelineLayoutCreateInfo = vks::initializers::pipelineLayoutCreateInfo(&descriptorSetLayouts.skysphere, 1);
		pipelineLayouts.skysphere = device.createPipelineLayout(pipelineLayoutCreateInfo);
	}

	void setupDescriptorSets()
	{
		vk::DescriptorSetAllocateInfo allocInfo;
		std::vector<vk::WriteDescriptorSet> writeDescriptorSets;

		// Terrain
		allocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorPool, &descriptorSetLayouts.terrain, 1);
		descriptorSets.terrain = device.allocateDescriptorSets(allocInfo)[0];

		writeDescriptorSets =
		{
			// Binding 0 : Shared tessellation shader ubo
			vks::initializers::writeDescriptorSet(
				descriptorSets.terrain, 
				vk::DescriptorType::eUniformBuffer, 
				0, 
				&uniformBuffers.terrainTessellation.descriptor),
			// Binding 1 : Displacement map
			vks::initializers::writeDescriptorSet(
				descriptorSets.terrain,
				vk::DescriptorType::eCombinedImageSampler,
				1,
				&textures.heightMap.descriptor),
			// Binding 2 : Color map (alpha channel)
			vks::initializers::writeDescriptorSet(
				descriptorSets.terrain,
				vk::DescriptorType::eCombinedImageSampler,
				2,
				&textures.terrainArray.descriptor),
		};
		device.updateDescriptorSets(writeDescriptorSets, nullptr);

		// Skysphere
		allocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorPool, &descriptorSetLayouts.skysphere, 1);
		descriptorSets.skysphere = device.allocateDescriptorSets(allocInfo)[0];

		writeDescriptorSets =
		{
			// Binding 0 : Vertex shader ubo
			vks::initializers::writeDescriptorSet(
				descriptorSets.skysphere,
				vk::DescriptorType::eUniformBuffer,
				0,
				&uniformBuffers.skysphereVertex.descriptor),
			// Binding 1 : Fragment shader color map
			vks::initializers::writeDescriptorSet(
				descriptorSets.skysphere,
				vk::DescriptorType::eCombinedImageSampler,
				1,
				&textures.skySphere.descriptor),
		};
		device.updateDescriptorSets(writeDescriptorSets, nullptr);
	}

	void preparePipelines()
	{
		vk::PipelineInputAssemblyStateCreateInfo inputAssemblyState =
			vks::initializers::pipelineInputAssemblyStateCreateInfo(
				vk::PrimitiveTopology::ePatchList,
				vk::PipelineInputAssemblyStateCreateFlags(),
				VK_FALSE);

		vk::PipelineRasterizationStateCreateInfo rasterizationState =
			vks::initializers::pipelineRasterizationStateCreateInfo(
				vk::PolygonMode::eFill,
				vk::CullModeFlagBits::eBack,
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
			vk::DynamicState::eScissor,
			vk::DynamicState::eLineWidth
		};
		vk::PipelineDynamicStateCreateInfo dynamicState =
			vks::initializers::pipelineDynamicStateCreateInfo(
				dynamicStateEnables);

		// We render the terrain as a grid of quad patches
		vk::PipelineTessellationStateCreateInfo tessellationState =
			vks::initializers::pipelineTessellationStateCreateInfo(4);

		std::array<vk::PipelineShaderStageCreateInfo, 4> shaderStages;

		// Terrain tessellation pipeline
		shaderStages[0] = loadShader(getAssetPath() + "shaders/terrain.vert.spv", vk::ShaderStageFlagBits::eVertex);
		shaderStages[1] = loadShader(getAssetPath() + "shaders/terrain.frag.spv", vk::ShaderStageFlagBits::eFragment);
		shaderStages[2] = loadShader(getAssetPath() + "shaders/terrain.tesc.spv", vk::ShaderStageFlagBits::eTessellationControl);
		shaderStages[3] = loadShader(getAssetPath() + "shaders/terrain.tese.spv", vk::ShaderStageFlagBits::eTessellationEvaluation);

		vk::GraphicsPipelineCreateInfo pipelineCreateInfo =
			vks::initializers::pipelineCreateInfo(
				pipelineLayouts.terrain,
				renderPass);

		pipelineCreateInfo.pVertexInputState = &vertices.inputState;
		pipelineCreateInfo.pInputAssemblyState = &inputAssemblyState;
		pipelineCreateInfo.pRasterizationState = &rasterizationState;
		pipelineCreateInfo.pColorBlendState = &colorBlendState;
		pipelineCreateInfo.pMultisampleState = &multisampleState;
		pipelineCreateInfo.pViewportState = &viewportState;
		pipelineCreateInfo.pDepthStencilState = &depthStencilState;
		pipelineCreateInfo.pDynamicState = &dynamicState;
		pipelineCreateInfo.pTessellationState = &tessellationState;
		pipelineCreateInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
		pipelineCreateInfo.pStages = shaderStages.data();
		pipelineCreateInfo.renderPass = renderPass;

		pipelines.terrain = device.createGraphicsPipelines(pipelineCache, pipelineCreateInfo)[0];

		// Terrain wireframe pipeline
		if (deviceFeatures.fillModeNonSolid) {
			rasterizationState.polygonMode = vk::PolygonMode::eLine;
			pipelines.wireframe = device.createGraphicsPipelines(pipelineCache, pipelineCreateInfo)[0];
		};

		// Skysphere pipeline
		rasterizationState.polygonMode = vk::PolygonMode::eFill;
		// Revert to triangle list topology
		inputAssemblyState.topology = vk::PrimitiveTopology::eTriangleList;
		// Reset tessellation state
		pipelineCreateInfo.pTessellationState = nullptr;
		// Don't write to depth buffer
		depthStencilState.depthWriteEnable = VK_FALSE;
		pipelineCreateInfo.stageCount = 2;
		pipelineCreateInfo.layout = pipelineLayouts.skysphere;
		shaderStages[0] = loadShader(getAssetPath() + "shaders/skysphere.vert.spv", vk::ShaderStageFlagBits::eVertex);
		shaderStages[1] = loadShader(getAssetPath() + "shaders/skysphere.frag.spv", vk::ShaderStageFlagBits::eFragment);
		pipelines.skysphere = device.createGraphicsPipelines(pipelineCache, pipelineCreateInfo)[0];
	}

	// Prepare and initialize uniform buffer containing shader uniforms
	void prepareUniformBuffers()
	{
		// Shared tessellation shader stages uniform buffer
		vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eUniformBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			&uniformBuffers.terrainTessellation,
			sizeof(uboTess));

		// Skysphere vertex shader uniform buffer
		vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eUniformBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			&uniformBuffers.skysphereVertex,
			sizeof(uboVS));

		// Map persistent
		uniformBuffers.terrainTessellation.map();
		uniformBuffers.skysphereVertex.map();

		updateUniformBuffers();
	}

	void updateUniformBuffers()
	{
		// Tessellation

		uboTess.projection = camera.matrices.perspective;
		uboTess.modelview = camera.matrices.view * glm::mat4();
		uboTess.lightPos.y = -0.5f - uboTess.displacementFactor; // todo: Not uesed yet
		uboTess.viewportDim = glm::vec2((float)width, (float)height);

		frustum.update(uboTess.projection * uboTess.modelview);
		memcpy(uboTess.frustumPlanes, frustum.planes.data(), sizeof(glm::vec4) * 6);

		float savedFactor = uboTess.tessellationFactor;
		if (!tessellation)
		{
			// Setting this to zero sets all tessellation factors to 1.0 in the shader
			uboTess.tessellationFactor = 0.0f;
		}

		memcpy(uniformBuffers.terrainTessellation.mapped, &uboTess, sizeof(uboTess));

		if (!tessellation)
		{
			uboTess.tessellationFactor = savedFactor;
		}

		// Skysphere vertex shader
		uboVS.mvp = camera.matrices.perspective * glm::mat4(glm::mat3(camera.matrices.view));
		memcpy(uniformBuffers.skysphereVertex.mapped, &uboVS, sizeof(uboVS));
	}

	void draw()
	{
		VulkanExampleBase::prepareFrame();

		// Command buffer to be sumitted to the queue
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &drawCmdBuffers[currentBuffer];

		// Submit to queue
		queue.submit(submitInfo, vk::Fence(nullptr));

		if (deviceFeatures.pipelineStatisticsQuery) {
			// Read query results for displaying in next frame
			getQueryResults();
		}

		VulkanExampleBase::submitFrame();
	}

	void prepare()
	{
		// Check if device supports tessellation shaders
		if (!deviceFeatures.tessellationShader)
		{
			vks::tools::exitFatal("Selected GPU does not support tessellation shaders!", "Feature not supported");
		}

		VulkanExampleBase::prepare();
		loadAssets();
		generateTerrain();
		if (deviceFeatures.pipelineStatisticsQuery) {
			setupQueryResultBuffer();
		}
		setupVertexDescriptions();
		prepareUniformBuffers();
		setupDescriptorSetLayouts();
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

	void changeTessellationFactor(float delta)
	{
		uboTess.tessellationFactor += delta;
		uboTess.tessellationFactor = fmax(0.25f, fmin(uboTess.tessellationFactor, 4.0f));
		updateUniformBuffers();
		updateTextOverlay();
	}

	void toggleWireframe()
	{
		wireframe = !wireframe;
		reBuildCommandBuffers();
		updateUniformBuffers();
	}

	void toggleTessellation()
	{
		tessellation = !tessellation;
		updateUniformBuffers();
	}

	virtual void keyPressed(uint32_t keyCode)
	{
		switch (keyCode)
		{
		case KEY_KPADD:
		case GAMEPAD_BUTTON_R1:
			changeTessellationFactor(0.05f);
			break;
		case KEY_KPSUB:
		case GAMEPAD_BUTTON_L1:
			changeTessellationFactor(-0.05f);
			break;
		case KEY_F:
		case GAMEPAD_BUTTON_A:
			if (deviceFeatures.fillModeNonSolid) {
				toggleWireframe();
			}
			break;
		case KEY_T:
		case GAMEPAD_BUTTON_X:
			toggleTessellation();
			break;
		}
	}

	virtual void getOverlayText(VulkanTextOverlay *textOverlay)
	{
		std::stringstream ss;
		ss << std::setprecision(2) << std::fixed << uboTess.tessellationFactor;

#if defined(__ANDROID__)
		textOverlay->addText("Tessellation factor: " + ss.str() + " (Buttons L1/R1)", 5.0f, 85.0f, VulkanTextOverlay::alignLeft);
		textOverlay->addText("Press \"Button A\" to toggle wireframe", 5.0f, 100.0f, VulkanTextOverlay::alignLeft);
		textOverlay->addText("Press \"Button X\" to toggle tessellation", 5.0f, 115.0f, VulkanTextOverlay::alignLeft);
#else
		textOverlay->addText("Tessellation factor: " + ss.str() + " (numpad +/-)", 5.0f, 85.0f, VulkanTextOverlay::alignLeft);
		textOverlay->addText("Press \"f\" to toggle wireframe", 5.0f, 100.0f, VulkanTextOverlay::alignLeft);
		textOverlay->addText("Press \"t\" to toggle tessellation", 5.0f, 115.0f, VulkanTextOverlay::alignLeft);
#endif

		textOverlay->addText("pipeline stats:", width - 5.0f, 5.0f, VulkanTextOverlay::alignRight);
		textOverlay->addText("VS:" + std::to_string(pipelineStats[0]), width - 5.0f, 20.0f, VulkanTextOverlay::alignRight);
		textOverlay->addText("TE:" + std::to_string(pipelineStats[1]), width - 5.0f, 35.0f, VulkanTextOverlay::alignRight);
	}
};

VULKAN_EXAMPLE_MAIN()