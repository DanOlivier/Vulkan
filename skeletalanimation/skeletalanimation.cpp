/*
* Vulkan Example - Skeletal animation
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
#include <map>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <assimp/Importer.hpp> 
#include <assimp/scene.h>     
#include <assimp/postprocess.h>
#include <assimp/cimport.h>

#include <vulkan/vulkan.hpp>
#include "vulkanexamplebase.h"
#include "VulkanBuffer.hpp"
#include "VulkanTexture.hpp"
#include "VulkanModel.hpp"

#define VERTEX_BUFFER_BIND_ID 0
#define ENABLE_VALIDATION false

// Vertex layout used in this example
struct Vertex {
	glm::vec3 pos;
	glm::vec3 normal;
	glm::vec2 uv;
	glm::vec3 color;
	// Max. four bones per vertex
	float boneWeights[4];
	uint32_t boneIDs[4];
};

// Vertex layout for the models
vks::VertexLayout vertexLayout = vks::VertexLayout({
	vks::VERTEX_COMPONENT_POSITION,
	vks::VERTEX_COMPONENT_NORMAL,
	vks::VERTEX_COMPONENT_UV,
	vks::VERTEX_COMPONENT_COLOR,
	vks::VERTEX_COMPONENT_DUMMY_VEC4,
	vks::VERTEX_COMPONENT_DUMMY_VEC4,
});

// Maximum number of bones per mesh
// Must not be higher than same const in skinning shader
#define MAX_BONES 64
// Maximum number of bones per vertex
#define MAX_BONES_PER_VERTEX 4

// Skinned mesh class

// Per-vertex bone IDs and weights
struct VertexBoneData
{
	std::array<uint32_t, MAX_BONES_PER_VERTEX> IDs;
	std::array<float, MAX_BONES_PER_VERTEX> weights;

	// Ad bone weighting to vertex info
	void add(uint32_t boneID, float weight)
	{
		for (uint32_t i = 0; i < MAX_BONES_PER_VERTEX; i++)
		{
			if (weights[i] == 0.0f)
			{
				IDs[i] = boneID;
				weights[i] = weight;
				return;
			}
		}
	}
};

// Stores information on a single bone
struct BoneInfo
{
	aiMatrix4x4 offset;
	aiMatrix4x4 finalTransformation;

	BoneInfo()
	{
		offset = aiMatrix4x4();
		finalTransformation = aiMatrix4x4();
	};
};

class SkinnedMesh 
{
public:
	// Bone related stuff
	// Maps bone name with index
	std::map<std::string, uint32_t> boneMapping;
	// Bone details
	std::vector<BoneInfo> boneInfo;
	// Number of bones present
	uint32_t numBones = 0;
	// Root inverese transform matrix
	aiMatrix4x4 globalInverseTransform;
	// Per-vertex bone info
	std::vector<VertexBoneData> bones;
	// Bone transformations
	std::vector<aiMatrix4x4> boneTransforms;

	// Modifier for the animation 
	float animationSpeed = 0.75f;
	// Currently active animation
	aiAnimation* pAnimation;

	// Vulkan buffers
	vks::Model vertexBuffer;

	// Store reference to the ASSIMP scene for accessing properties of it during animation
	Assimp::Importer Importer;
	const aiScene* scene;

	// Set active animation by index
	void setAnimation(uint32_t animationIndex)
	{
		assert(animationIndex < scene->mNumAnimations);
		pAnimation = scene->mAnimations[animationIndex];
	}

	// Load bone information from ASSIMP mesh
	void loadBones(const aiMesh* pMesh, uint32_t vertexOffset, std::vector<VertexBoneData>& Bones)
	{
		for (uint32_t i = 0; i < pMesh->mNumBones; i++)
		{
			uint32_t index = 0;

			assert(pMesh->mNumBones <= MAX_BONES);

			std::string name(pMesh->mBones[i]->mName.data);

			if (boneMapping.find(name) == boneMapping.end())
			{
				// Bone not present, add new one
				index = numBones;
				numBones++;
				BoneInfo bone;
				boneInfo.push_back(bone);
				boneInfo[index].offset = pMesh->mBones[i]->mOffsetMatrix;
				boneMapping[name] = index;
			}
			else
			{
				index = boneMapping[name];
			}

			for (uint32_t j = 0; j < pMesh->mBones[i]->mNumWeights; j++)
			{
				uint32_t vertexID = vertexOffset + pMesh->mBones[i]->mWeights[j].mVertexId;
				Bones[vertexID].add(index, pMesh->mBones[i]->mWeights[j].mWeight);
			}
		}
		boneTransforms.resize(numBones);
	}

	// Recursive bone transformation for given animation time
	void update(float time)
	{
		float TicksPerSecond = (float)(scene->mAnimations[0]->mTicksPerSecond != 0 ? scene->mAnimations[0]->mTicksPerSecond : 25.0f);
		float TimeInTicks = time * TicksPerSecond;
		float AnimationTime = fmod(TimeInTicks, (float)scene->mAnimations[0]->mDuration);

		aiMatrix4x4 identity = aiMatrix4x4();
		readNodeHierarchy(AnimationTime, scene->mRootNode, identity);

		for (uint32_t i = 0; i < boneTransforms.size(); i++)
		{
			boneTransforms[i] = boneInfo[i].finalTransformation;
		}
	}

	~SkinnedMesh()
	{
		vertexBuffer.vertices.destroy();
		vertexBuffer.indices.destroy();
	}

private:
	// Find animation for a given node
	const aiNodeAnim* findNodeAnim(const aiAnimation* animation, const std::string nodeName)
	{
		for (uint32_t i = 0; i < animation->mNumChannels; i++)
		{
			const aiNodeAnim* nodeAnim = animation->mChannels[i];
			if (std::string(nodeAnim->mNodeName.data) == nodeName)
			{
				return nodeAnim;
			}
		}
		return nullptr;
	}

	// Returns a 4x4 matrix with interpolated translation between current and next frame
	aiMatrix4x4 interpolateTranslation(float time, const aiNodeAnim* pNodeAnim)
	{
		aiVector3D translation;

		if (pNodeAnim->mNumPositionKeys == 1)
		{
			translation = pNodeAnim->mPositionKeys[0].mValue;
		}
		else
		{
			uint32_t frameIndex = 0;
			for (uint32_t i = 0; i < pNodeAnim->mNumPositionKeys - 1; i++)
			{
				if (time < (float)pNodeAnim->mPositionKeys[i + 1].mTime)
				{
					frameIndex = i;
					break;
				}
			}

			aiVectorKey currentFrame = pNodeAnim->mPositionKeys[frameIndex];
			aiVectorKey nextFrame = pNodeAnim->mPositionKeys[(frameIndex + 1) % pNodeAnim->mNumPositionKeys];

			float delta = (time - (float)currentFrame.mTime) / (float)(nextFrame.mTime - currentFrame.mTime);

			const aiVector3D& start = currentFrame.mValue;
			const aiVector3D& end = nextFrame.mValue;

			translation = (start + delta * (end - start));
		}

		aiMatrix4x4 mat;
		aiMatrix4x4::Translation(translation, mat);
		return mat;
	}

	// Returns a 4x4 matrix with interpolated rotation between current and next frame
	aiMatrix4x4 interpolateRotation(float time, const aiNodeAnim* pNodeAnim)
	{
		aiQuaternion rotation;

		if (pNodeAnim->mNumRotationKeys == 1)
		{
			rotation = pNodeAnim->mRotationKeys[0].mValue;
		}
		else
		{
			uint32_t frameIndex = 0;
			for (uint32_t i = 0; i < pNodeAnim->mNumRotationKeys - 1; i++)
			{
				if (time < (float)pNodeAnim->mRotationKeys[i + 1].mTime)
				{
					frameIndex = i;
					break;
				}
			}

			aiQuatKey currentFrame = pNodeAnim->mRotationKeys[frameIndex];
			aiQuatKey nextFrame = pNodeAnim->mRotationKeys[(frameIndex + 1) % pNodeAnim->mNumRotationKeys];

			float delta = (time - (float)currentFrame.mTime) / (float)(nextFrame.mTime - currentFrame.mTime);

			const aiQuaternion& start = currentFrame.mValue;
			const aiQuaternion& end = nextFrame.mValue;

			aiQuaternion::Interpolate(rotation, start, end, delta);
			rotation.Normalize();
		}

		aiMatrix4x4 mat(rotation.GetMatrix());
		return mat;
	}


	// Returns a 4x4 matrix with interpolated scaling between current and next frame
	aiMatrix4x4 interpolateScale(float time, const aiNodeAnim* pNodeAnim)
	{
		aiVector3D scale;

		if (pNodeAnim->mNumScalingKeys == 1)
		{
			scale = pNodeAnim->mScalingKeys[0].mValue;
		}
		else
		{
			uint32_t frameIndex = 0;
			for (uint32_t i = 0; i < pNodeAnim->mNumScalingKeys - 1; i++)
			{
				if (time < (float)pNodeAnim->mScalingKeys[i + 1].mTime)
				{
					frameIndex = i;
					break;
				}
			}

			aiVectorKey currentFrame = pNodeAnim->mScalingKeys[frameIndex];
			aiVectorKey nextFrame = pNodeAnim->mScalingKeys[(frameIndex + 1) % pNodeAnim->mNumScalingKeys];

			float delta = (time - (float)currentFrame.mTime) / (float)(nextFrame.mTime - currentFrame.mTime);

			const aiVector3D& start = currentFrame.mValue;
			const aiVector3D& end = nextFrame.mValue;

			scale = (start + delta * (end - start));
		}

		aiMatrix4x4 mat;
		aiMatrix4x4::Scaling(scale, mat);
		return mat;
	}

	// Get node hierarchy for current animation time
	void readNodeHierarchy(float AnimationTime, const aiNode* pNode, const aiMatrix4x4& ParentTransform)
	{
		std::string NodeName(pNode->mName.data);

		aiMatrix4x4 NodeTransformation(pNode->mTransformation);

		const aiNodeAnim* pNodeAnim = findNodeAnim(pAnimation, NodeName);

		if (pNodeAnim)
		{
			// Get interpolated matrices between current and next frame
			aiMatrix4x4 matScale = interpolateScale(AnimationTime, pNodeAnim);
			aiMatrix4x4 matRotation = interpolateRotation(AnimationTime, pNodeAnim);
			aiMatrix4x4 matTranslation = interpolateTranslation(AnimationTime, pNodeAnim);

			NodeTransformation = matTranslation * matRotation * matScale;
		}

		aiMatrix4x4 GlobalTransformation = ParentTransform * NodeTransformation;

		if (boneMapping.find(NodeName) != boneMapping.end())
		{
			uint32_t BoneIndex = boneMapping[NodeName];
			boneInfo[BoneIndex].finalTransformation = globalInverseTransform * GlobalTransformation * boneInfo[BoneIndex].offset;
		}

		for (uint32_t i = 0; i < pNode->mNumChildren; i++)
		{
			readNodeHierarchy(AnimationTime, pNode->mChildren[i], GlobalTransformation);
		}
	}
};

class VulkanExample : public VulkanExampleBase
{
public:
	struct {
		vks::Texture2D colorMap;
		vks::Texture2D floor;
	} textures;

	SkinnedMesh *skinnedMesh = nullptr;

	struct {
		vks::Buffer mesh;
		vks::Buffer floor;
	} uniformBuffers;

	struct {
		glm::mat4 projection;
		glm::mat4 model;
		glm::mat4 view;
		glm::mat4 bones[MAX_BONES];
		glm::vec4 lightPos = glm::vec4(0.0f, -250.0f, 250.0f, 1.0);
		glm::vec4 viewPos;
	} uboVS;

	struct {
		glm::mat4 projection;
		glm::mat4 model;
		glm::mat4 view;
		glm::vec4 lightPos = glm::vec4(0.0, 0.0f, -25.0f, 1.0);
		glm::vec4 viewPos;
		glm::vec2 uvOffset;
	} uboFloor;

	struct {
		vk::Pipeline skinning;
		vk::Pipeline texture;
	} pipelines;

	struct {
		vks::Model floor;
	} models;

	vk::PipelineLayout pipelineLayout;
	vk::DescriptorSet descriptorSet;
	vk::DescriptorSetLayout descriptorSetLayout;

	struct {
		vk::DescriptorSet skinning;
		vk::DescriptorSet floor;
	} descriptorSets;

	float runningTime = 0.0f;

	VulkanExample() : VulkanExampleBase(ENABLE_VALIDATION)
	{
		zoom = -150.0f;
		zoomSpeed = 2.5f;
		rotationSpeed = 0.5f;
		rotation = { -182.5f, -38.5f, 180.0f };
		enableTextOverlay = true;
		title = "Vulkan Example - Skeletal animation";
		cameraPos = { 0.0f, 0.0f, 12.0f };
	}

	~VulkanExample()
	{
		// Clean up used Vulkan resources 
		// Note : Inherited destructor cleans up resources stored in base class
		device.destroyPipeline(pipelines.skinning);
		device.destroyPipeline(pipelines.texture);

		device.destroyPipelineLayout(pipelineLayout);
		device.destroyDescriptorSetLayout(descriptorSetLayout);

		textures.colorMap.destroy();
		textures.floor.destroy();

		uniformBuffers.mesh.destroy();
		uniformBuffers.floor.destroy();

		models.floor.destroy();
		delete(skinnedMesh);
	}

	void buildCommandBuffers()
	{
		vk::CommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();

		vk::ClearValue clearValues[2];
		clearValues[0].color = { { 0.0f, 0.0f, 0.0f, 0.0f} };
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
			renderPassBeginInfo.framebuffer = frameBuffers[i];

			drawCmdBuffers[i].begin(cmdBufInfo);

			drawCmdBuffers[i].beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);

			vk::Viewport viewport = vks::initializers::viewport((float)width, (float)height, 0.0f, 1.0f);
			drawCmdBuffers[i].setViewport(0, viewport);

			vk::Rect2D scissor = vks::initializers::rect2D(width, height, 0, 0);
			vkCmdSetScissor(drawCmdBuffers[i], 0, 1, &scissor);

			vk::DeviceSize offsets[1] = { 0 };

			// Skinned mesh
			drawCmdBuffers[i].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSet, nullptr);
			drawCmdBuffers[i].bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.skinning);

			drawCmdBuffers[i].bindVertexBuffers(VERTEX_BUFFER_BIND_ID, skinnedMesh->vertexBuffer.vertices.buffer, offsets);
			drawCmdBuffers[i].bindIndexBuffer(skinnedMesh->vertexBuffer.indices.buffer, 0, vk::IndexType::eUint32);
			drawCmdBuffers[i].drawIndexed(skinnedMesh->vertexBuffer.indexCount, 1, 0, 0, 0);

			// Floor
			drawCmdBuffers[i].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSets.floor, nullptr);
			drawCmdBuffers[i].bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.texture);

			drawCmdBuffers[i].bindVertexBuffers(VERTEX_BUFFER_BIND_ID, models.floor.vertices.buffer, offsets);
			drawCmdBuffers[i].bindIndexBuffer(models.floor.indices.buffer, 0, vk::IndexType::eUint32);
			drawCmdBuffers[i].drawIndexed(models.floor.indexCount, 1, 0, 0, 0);

			drawCmdBuffers[i].endRenderPass();

			drawCmdBuffers[i].end();
		}
	}

	// Load a mesh based on data read via assimp 
	void loadMesh()
	{
		skinnedMesh = new SkinnedMesh();

		std::string filename = getAssetPath() + "models/goblin.dae";

#if defined(__ANDROID__)
		// Meshes are stored inside the apk on Android (compressed)
		// So they need to be loaded via the asset manager

		AAsset* asset = AAssetManager_open(androidApp->activity->assetManager, filename.c_str(), AASSET_MODE_STREAMING);
		assert(asset);
		size_t size = AAsset_getLength(asset);

		assert(size > 0);

		void *meshData = malloc(size);
		AAsset_read(asset, meshData, size);
		AAsset_close(asset);

		skinnedMesh->scene = skinnedMesh->Importer.ReadFileFromMemory(meshData, size, 0);

		free(meshData);
#else
		skinnedMesh->scene = skinnedMesh->Importer.ReadFile(filename.c_str(), 0);
#endif
		skinnedMesh->setAnimation(0);

		// Setup bones
		// One vertex bone info structure per vertex
		uint32_t vertexCount(0);		
		for (uint32_t m = 0; m < skinnedMesh->scene->mNumMeshes; m++) {
			vertexCount += skinnedMesh->scene->mMeshes[m]->mNumVertices;
		};
		skinnedMesh->bones.resize(vertexCount);
		// Store global inverse transform matrix of root node 
		skinnedMesh->globalInverseTransform = skinnedMesh->scene->mRootNode->mTransformation;
		skinnedMesh->globalInverseTransform.Inverse();
		// Load bones (weights and IDs)
		uint32_t vertexBase(0);
		for (uint32_t m = 0; m < skinnedMesh->scene->mNumMeshes; m++) {
			aiMesh *paiMesh = skinnedMesh->scene->mMeshes[m];
			if (paiMesh->mNumBones > 0) {
				skinnedMesh->loadBones(paiMesh, vertexBase, skinnedMesh->bones);
			}
			vertexBase += skinnedMesh->scene->mMeshes[m]->mNumVertices;
		}

		// Generate vertex buffer
		std::vector<Vertex> vertexBuffer;
		// Iterate through all meshes in the file and extract the vertex information used in this demo
		vertexBase = 0;
		for (uint32_t m = 0; m < skinnedMesh->scene->mNumMeshes; m++) {
			for (uint32_t v = 0; v < skinnedMesh->scene->mMeshes[m]->mNumVertices; v++) {
				Vertex vertex;

				vertex.pos = glm::make_vec3(&skinnedMesh->scene->mMeshes[m]->mVertices[v].x);
				vertex.normal = glm::make_vec3(&skinnedMesh->scene->mMeshes[m]->mNormals[v].x);
				vertex.uv = glm::make_vec2(&skinnedMesh->scene->mMeshes[m]->mTextureCoords[0][v].x);
				vertex.color = (skinnedMesh->scene->mMeshes[m]->HasVertexColors(0)) ? glm::make_vec3(&skinnedMesh->scene->mMeshes[m]->mColors[0][v].r) : glm::vec3(1.0f);

				// Fetch bone weights and IDs
				for (uint32_t j = 0; j < MAX_BONES_PER_VERTEX; j++) {
					vertex.boneWeights[j] = skinnedMesh->bones[vertexBase + v].weights[j];
					vertex.boneIDs[j] = skinnedMesh->bones[vertexBase + v].IDs[j];
				}

				vertexBuffer.push_back(vertex);
			}
			vertexBase += skinnedMesh->scene->mMeshes[m]->mNumVertices;
		}
		vk::DeviceSize vertexBufferSize = vertexBuffer.size() * sizeof(Vertex);

		// Generate index buffer from loaded mesh file
		std::vector<uint32_t> indexBuffer;
		for (uint32_t m = 0; m < skinnedMesh->scene->mNumMeshes; m++) {
			uint32_t indexBase = static_cast<uint32_t>(indexBuffer.size());
			for (uint32_t f = 0; f < skinnedMesh->scene->mMeshes[m]->mNumFaces; f++) {
				for (uint32_t i = 0; i < 3; i++)
				{
					indexBuffer.push_back(skinnedMesh->scene->mMeshes[m]->mFaces[f].mIndices[i] + indexBase);
				}
			}
		}
		vk::DeviceSize indexBufferSize = indexBuffer.size() * sizeof(uint32_t);
		skinnedMesh->vertexBuffer.indexCount = static_cast<uint32_t>(indexBuffer.size());

		struct {
			vk::Buffer buffer;
			vk::DeviceMemory memory;
		} vertexStaging, indexStaging;

		// Create staging buffers
		// Vertex data
		VK_CHECK_RESULT(vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eTransferSrc,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			vertexBufferSize,
			&vertexStaging.buffer,
			&vertexStaging.memory,
			vertexBuffer.data()));
		// Index data
		VK_CHECK_RESULT(vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eTransferSrc,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			indexBufferSize,
			&indexStaging.buffer,
			&indexStaging.memory,
			indexBuffer.data()));

		// Create device local buffers
		// Vertex buffer
		VK_CHECK_RESULT(vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst,
			vk::MemoryPropertyFlagBits::eDeviceLocal,
			&skinnedMesh->vertexBuffer.vertices,
			vertexBufferSize));
		// Index buffer
		VK_CHECK_RESULT(vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst,
			vk::MemoryPropertyFlagBits::eDeviceLocal,
			&skinnedMesh->vertexBuffer.indices,
			indexBufferSize));

		// Copy from staging buffers
		vk::CommandBuffer copyCmd = VulkanExampleBase::createCommandBuffer(vk::CommandBufferLevel::ePrimary, true);

		vk::BufferCopy copyRegion = {};

		copyRegion.size = vertexBufferSize;
		copyCmd.copyBuffer(
			vertexStaging.buffer,
			skinnedMesh->vertexBuffer.vertices.buffer,
			copyRegion);

		copyRegion.size = indexBufferSize;
		copyCmd.copyBuffer(
			indexStaging.buffer,
			skinnedMesh->vertexBuffer.indices.buffer,
			copyRegion);

		VulkanExampleBase::flushCommandBuffer(copyCmd, queue, true);

		device.destroyBuffer(vertexStaging.buffer);
		device.freeMemory(vertexStaging.memory);
		device.destroyBuffer(indexStaging.buffer);
		device.freeMemory(indexStaging.memory);
	}

	void loadAssets()
	{
		models.floor.loadFromFile(getAssetPath() + "models/plane_z.obj", vertexLayout, 512.0f, vulkanDevice, queue);

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

		textures.colorMap.loadFromFile(getAssetPath() + "textures/goblin" + texFormatSuffix + ".ktx", texFormat, vulkanDevice, queue);
		textures.floor.loadFromFile(getAssetPath() + "textures/trail" + texFormatSuffix + ".ktx", texFormat, vulkanDevice, queue);
	}

	void setupDescriptorPool()
	{
		// Example uses one ubo and one combined image sampler
		std::vector<vk::DescriptorPoolSize> poolSizes =
		{
			vks::initializers::descriptorPoolSize(vk::DescriptorType::eUniformBuffer, 2),
			vks::initializers::descriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, 2),
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
			// Binding 1 : Fragment shader combined sampler
			vks::initializers::descriptorSetLayoutBinding(
				vk::DescriptorType::eCombinedImageSampler,
				vk::ShaderStageFlagBits::eFragment,
				1),
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
		
		vk::DescriptorImageInfo texDescriptor =
			vks::initializers::descriptorImageInfo(
				textures.colorMap.sampler,
				textures.colorMap.view,
				vk::ImageLayout::eGeneral);

		std::vector<vk::WriteDescriptorSet> writeDescriptorSets =
		{
			// Binding 0 : Vertex shader uniform buffer
			vks::initializers::writeDescriptorSet(
				descriptorSet,
				vk::DescriptorType::eUniformBuffer,
				0,
				&uniformBuffers.mesh.descriptor),
			// Binding 1 : Color map 
			vks::initializers::writeDescriptorSet(
				descriptorSet,
				vk::DescriptorType::eCombinedImageSampler,
				1,
				&texDescriptor)
		};

		vkUpdateDescriptorSets(device, writeDescriptorSets.size(), writeDescriptorSets.data(), 0, NULL);

		// Floor
		descriptorSets.floor = device.allocateDescriptorSets(allocInfo)[0];

		texDescriptor.imageView = textures.floor.view;
		texDescriptor.sampler = textures.floor.sampler;

		writeDescriptorSets.clear();

		// Binding 0 : Vertex shader uniform buffer
		writeDescriptorSets.push_back(
			vks::initializers::writeDescriptorSet(
				descriptorSets.floor,
				vk::DescriptorType::eUniformBuffer,
				0,
				&uniformBuffers.floor.descriptor));
		// Binding 1 : Color map 
		writeDescriptorSets.push_back(
			vks::initializers::writeDescriptorSet(
				descriptorSets.floor,
				vk::DescriptorType::eCombinedImageSampler,
				1,
				&texDescriptor));

		vkUpdateDescriptorSets(device, writeDescriptorSets.size(), writeDescriptorSets.data(), 0, NULL);
	}

	void preparePipelines()
	{
		vk::PipelineInputAssemblyStateCreateInfo inputAssemblyState =
			vks::initializers::pipelineInputAssemblyStateCreateInfo(
				vk::PrimitiveTopology::eTriangleList,
				0,
				VK_FALSE);

		vk::PipelineRasterizationStateCreateInfo rasterizationState =
			vks::initializers::pipelineRasterizationStateCreateInfo(
				vk::PolygonMode::eFill,
				vk::CullModeFlagBits::eBack,
				vk::FrontFace::eClockwise,
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
				VK_TRUE,
				VK_TRUE,
				vk::CompareOp::eLessOrEqual);

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
				dynamicStateEnables.size(),
				0);

		// Skinned rendering pipeline
		std::array<vk::PipelineShaderStageCreateInfo, 2> shaderStages;

		vk::GraphicsPipelineCreateInfo pipelineCreateInfo =
			vks::initializers::pipelineCreateInfo(
				pipelineLayout,
				renderPass,
				0);

		pipelineCreateInfo.pInputAssemblyState = &inputAssemblyState;
		pipelineCreateInfo.pRasterizationState = &rasterizationState;
		pipelineCreateInfo.pColorBlendState = &colorBlendState;
		pipelineCreateInfo.pMultisampleState = &multisampleState;
		pipelineCreateInfo.pViewportState = &viewportState;
		pipelineCreateInfo.pDepthStencilState = &depthStencilState;
		pipelineCreateInfo.pDynamicState = &dynamicState;
		pipelineCreateInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
		pipelineCreateInfo.pStages = shaderStages.data();

		// Shared vertex inputs

		// Binding description
		vk::VertexInputBindingDescription vertexInputBinding =
			vks::initializers::vertexInputBindingDescription(VERTEX_BUFFER_BIND_ID, sizeof(Vertex), vk::VertexInputRate::eVertex);

		// Attribute descriptions
		// Describes memory layout and shader positions
		std::vector<vk::VertexInputAttributeDescription> vertexInputAttributes = {
			vks::initializers::vertexInputAttributeDescription(VERTEX_BUFFER_BIND_ID, 0, vk::Format::eR32G32B32Sfloat, 0),						// Location 0: Position		
			vks::initializers::vertexInputAttributeDescription(VERTEX_BUFFER_BIND_ID, 1, vk::Format::eR32G32B32Sfloat, sizeof(float) * 3),		// Location 1: Normal		
			vks::initializers::vertexInputAttributeDescription(VERTEX_BUFFER_BIND_ID, 2, vk::Format::eR32G32Sfloat, sizeof(float) * 6),			// Location 2: Texture coordinates		
			vks::initializers::vertexInputAttributeDescription(VERTEX_BUFFER_BIND_ID, 3, vk::Format::eR32G32B32Sfloat, sizeof(float) * 8),		// Location 3: Color		
			vks::initializers::vertexInputAttributeDescription(VERTEX_BUFFER_BIND_ID, 4, vk::Format::eR32G32B32A32Sfloat, sizeof(float) * 11),	// Location 4: Bone weights		
			vks::initializers::vertexInputAttributeDescription(VERTEX_BUFFER_BIND_ID, 5, vk::Format::eR32G32B32A32Sint, sizeof(float) * 15),		// Location 5: Bone IDs
		};

		vk::PipelineVertexInputStateCreateInfo vertexInputState = vks::initializers::pipelineVertexInputStateCreateInfo();
		vertexInputState.vertexBindingDescriptionCount = 1;
		vertexInputState.pVertexBindingDescriptions = &vertexInputBinding;
		vertexInputState.vertexAttributeDescriptionCount = static_cast<uint32_t>(vertexInputAttributes.size());
		vertexInputState.pVertexAttributeDescriptions = vertexInputAttributes.data();

		pipelineCreateInfo.pVertexInputState = &vertexInputState;

		// Skinned mesh rendering pipeline
		shaderStages[0] = loadShader(getAssetPath() + "shaders/skeletalanimation/mesh.vert.spv", vk::ShaderStageFlagBits::eVertex);
		shaderStages[1] = loadShader(getAssetPath() + "shaders/skeletalanimation/mesh.frag.spv", vk::ShaderStageFlagBits::eFragment);
		pipelines.skinning = device.createGraphicsPipelines(pipelineCache, pipelineCreateInfo)[0];

		// Environment rendering pipeline
		shaderStages[0] = loadShader(getAssetPath() + "shaders/skeletalanimation/texture.vert.spv", vk::ShaderStageFlagBits::eVertex);
		shaderStages[1] = loadShader(getAssetPath() + "shaders/skeletalanimation/texture.frag.spv", vk::ShaderStageFlagBits::eFragment);
		pipelines.texture = device.createGraphicsPipelines(pipelineCache, pipelineCreateInfo)[0];
	}

	// Prepare and initialize uniform buffer containing shader uniforms
	void prepareUniformBuffers()
	{
		// Mesh uniform buffer block
		vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eUniformBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			&uniformBuffers.mesh,
			sizeof(uboVS));
		// Map persistant
		uniformBuffers.mesh.map();

		// Floor uniform buffer block
		vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eUniformBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			&uniformBuffers.floor,
			sizeof(uboFloor));
		// Map persistant
		uniformBuffers.floor.map();

		updateUniformBuffers(true);
	}

	void updateUniformBuffers(bool viewChanged)
	{
		if (viewChanged)
		{
			uboVS.projection = glm::perspective(glm::radians(60.0f), (float)width / (float)height, 0.1f, 1024.0f);

			glm::mat4 viewMatrix = glm::translate(glm::mat4(), glm::vec3(0.0f, 0.0f, zoom));
			viewMatrix = glm::rotate(viewMatrix, glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
			viewMatrix = glm::scale(viewMatrix, glm::vec3(0.025f));

			uboVS.view = viewMatrix * glm::translate(glm::mat4(), glm::vec3(cameraPos.x, -cameraPos.z, cameraPos.y) * 100.0f);
			uboVS.view = glm::rotate(uboVS.view, glm::radians(rotation.x), glm::vec3(1.0f, 0.0f, 0.0f));
			uboVS.view = glm::rotate(uboVS.view, glm::radians(rotation.z), glm::vec3(0.0f, 1.0f, 0.0f));
			uboVS.view = glm::rotate(uboVS.view, glm::radians(-rotation.y), glm::vec3(0.0f, 0.0f, 1.0f));

			uboVS.viewPos = glm::vec4(0.0f, 0.0f, -zoom, 0.0f);

			uboFloor.projection = uboVS.projection;
			uboFloor.view = viewMatrix;
			uboFloor.model = glm::translate(glm::mat4(), glm::vec3(cameraPos.x, -cameraPos.z, cameraPos.y) * 100.0f);
			uboFloor.model = glm::rotate(uboFloor.model, glm::radians(rotation.x), glm::vec3(1.0f, 0.0f, 0.0f));
			uboFloor.model = glm::rotate(uboFloor.model, glm::radians(rotation.z), glm::vec3(0.0f, 1.0f, 0.0f));
			uboFloor.model = glm::rotate(uboFloor.model, glm::radians(-rotation.y), glm::vec3(0.0f, 0.0f, 1.0f));
			uboFloor.model = glm::translate(uboFloor.model, glm::vec3(0.0f, 0.0f, -1800.0f));
			uboFloor.viewPos = glm::vec4(0.0f, 0.0f, -zoom, 0.0f);
		}

		// Update bones
		skinnedMesh->update(runningTime);
		for (uint32_t i = 0; i < skinnedMesh->boneTransforms.size(); i++)
		{
			uboVS.bones[i] = glm::transpose(glm::make_mat4(&skinnedMesh->boneTransforms[i].a1));
		}

		uniformBuffers.mesh.copyTo(&uboVS, sizeof(uboVS));

		// Update floor animation
		uboFloor.uvOffset.t -= 0.25f * skinnedMesh->animationSpeed * frameTimer;
		uniformBuffers.floor.copyTo(&uboFloor, sizeof(uboFloor));
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
		loadMesh();
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
			runningTime += frameTimer * skinnedMesh->animationSpeed;
			vkDeviceWaitIdle(device);
			updateUniformBuffers(false);
		}
	}

	virtual void viewChanged()
	{
		vkDeviceWaitIdle(device);
		updateUniformBuffers(true);
	}

	void changeAnimationSpeed(float delta)
	{
		skinnedMesh->animationSpeed += delta;
	}

	virtual void keyPressed(uint32_t keyCode)
	{
		switch (keyCode)
		{
		case KEY_KPADD:
		case GAMEPAD_BUTTON_R1:
			changeAnimationSpeed(0.1f);
			break;
		case KEY_KPSUB:
		case GAMEPAD_BUTTON_L1:
			changeAnimationSpeed(-0.1f);
			break;
		}
	}

	virtual void getOverlayText(VulkanTextOverlay *textOverlay)
	{
		if (skinnedMesh != nullptr)
		{
			std::stringstream ss;
			ss << std::setprecision(2) << std::fixed << skinnedMesh->animationSpeed;
#if defined(__ANDROID__)
			textOverlay->addText("Animation speed: " + ss.str() + " (Buttons L1/R1 to change)", 5.0f, 85.0f, VulkanTextOverlay::alignLeft);
#else
			textOverlay->addText("Animation speed: " + ss.str() + " (numpad +/- to change)", 5.0f, 85.0f, VulkanTextOverlay::alignLeft);
#endif
		}
	}
};

VULKAN_EXAMPLE_MAIN()