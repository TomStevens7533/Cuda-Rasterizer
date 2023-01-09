#include "ChunkBuilder.h"
#include <array>
#include <iostream>
#include <cmath>
#include <gtc/integer.hpp>

const std::array<float, 12> xFace2{
	0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1,
};
const std::array<float, 12> frontFace{
	0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1,
};

const std::array<float, 12> backFace{
	1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0,
};

const std::array<float, 12> leftFace{
	0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0,
};

const std::array<float, 12> rightFace{
	1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1,
};

const std::array<float, 12> topFace{
	0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0,
};
const std::array<float, 12> bottomFace{
	0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1
};

ChunkMesh::ChunkMesh(glm::vec3 originPos, float lodDistance) 
	: m_ThreadOriginPos{originPos}, m_ThreadLODdistance{lodDistance}
{
	m_StarterNode = new SVOInnerNode();
	m_StarterNode->pParentNode = nullptr;
	FillSVO();
	m_LODUpdatingThread = std::jthread{ &ChunkMesh::TraverseSVO, this };
}

ChunkMesh::~ChunkMesh()
{
	m_IsThreadRunning = false;
	while (m_SwapDone == false)
	{

	}
	if (m_StarterNode != nullptr) {
		delete m_StarterNode;
	}
}

void ChunkMesh::TraverseSVO()
{
	while (m_IsThreadRunning)
	{
		if (m_SwapDone == false) {
			std::cout << "Starting vertex/index creatiom" << std::endl;

			TraverseSVONode(m_StarterNode, CHUNKSIZE_X / 2, glm::vec3{ CHUNKSIZE_X / 2, CHUNKSIZE_X / 2 , CHUNKSIZE_X / 2 },
				m_ThreadOriginPos, m_ThreadLODdistance);

			std::cout << "Amount of block filled: " << m_blockdetected << std::endl;
			m_SwapDone = true;
			m_blockdetected = 0;
		}
	}

}
void ChunkMesh::SwapBuffers(glm::vec3 originPos, float lodDistance)
{
	if (m_SwapDone) {
		//Swap Buffers
		m_Vertices.clear();
		m_Indices.clear();
		m_IndicesIndex = 0;
		m_Facedetected = 0;

		m_Vertices = (m_ThreadVertices);
		m_Indices = (m_ThreadIndices);

		m_ThreadIndices.clear();
		m_ThreadVertices.clear();

		m_ThreadOriginPos = originPos;
		m_ThreadLODdistance = lodDistance;
		m_SwapDone = false;


	}
}
void ChunkMesh::TraverseSVONode(SVOBaseNode* pNode, int resolution, glm::vec3 nodeLocalPosition,
	glm::vec3 originPos, float lodDistance)
{


	if (pNode == nullptr)
		return;

	//If it is and end node stop here all data is further is sparse
	if (pNode->m_IsEndNode)
		return;

	if (dynamic_cast<SVOLeafNode*>(pNode)) {
		//FILL IN vertex information
		auto leafNode = static_cast<SVOLeafNode*>(pNode);
		if (leafNode->blockID != AIR) {
			++m_blockdetected;
			CheckGenerationOfFace(Faces::TOP,	leafNode, nodeLocalPosition);
			CheckGenerationOfFace(Faces::BOT,	leafNode, nodeLocalPosition);
			CheckGenerationOfFace(Faces::FRONT, leafNode, nodeLocalPosition);
			CheckGenerationOfFace(Faces::BACK,	leafNode, nodeLocalPosition);
			CheckGenerationOfFace(Faces::LEFT,	leafNode, nodeLocalPosition);
			CheckGenerationOfFace(Faces::RIGHT, leafNode, nodeLocalPosition);
		}
		return; //THIS NODE IS AN END NODE
	}
	SVOInnerNode* innerNode = static_cast<SVOInnerNode*>(pNode);

		int newChildResolution = resolution * 0.5f;
		for (size_t x = 0; x < 2; x++)
		{
			for (size_t y = 0; y < 2; y++)
			{
				for (size_t z = 0; z < 2; z++)
				{
					glm::vec3 newNodePosition = nodeLocalPosition;
					if (resolution > 1) {
						newNodePosition.x += (x == 0 ? -1 : 1) * newChildResolution;
						newNodePosition.y += (y == 0 ? -1 : 1) * newChildResolution;
						newNodePosition.z += (z == 0 ? -1 : 1) * newChildResolution;
					}
					else {
						newNodePosition.x += (x == 0 ? -1 : 0);
						newNodePosition.y += (y == 0 ? -1 : 0);
						newNodePosition.z += (z == 0 ? -1 : 0);
					}
			

					float distanceToCam = glm::distance(newNodePosition, originPos);
					int resolutionLODlevel = glm::log2(newChildResolution);
					int distanceLODLevel = (distanceToCam / lodDistance);
					int clampedDistanceLodLevel = glm::clamp(distanceLODLevel, 1, MAX_LEVEL);
			


					if (resolutionLODlevel <= MAX_LEVEL && resolutionLODlevel == clampedDistanceLodLevel && distanceLODLevel >= 1)
					{

						//DRAW LOD LEVEL
						std::pair<bool, bool> blockPair;
						blockPair.first = false;
						blockPair.second = false;
						HasMultipleBlocks(innerNode->children[x][y][z], blockPair);
						if (blockPair.first == true && blockPair.second == true) {
							//Has multiple blocktypes RENDER
							glm::vec3 nodePos = newNodePosition;


							nodePos.x += newChildResolution;
							nodePos.y -= newChildResolution;
							nodePos.z -= newChildResolution;

							GenerateFace(Faces::BACK, nodePos,  resolution);
							GenerateFace(Faces::BOT, nodePos,	resolution);
							GenerateFace(Faces::FRONT, nodePos, resolution);
							GenerateFace(Faces::LEFT, nodePos,  resolution);
							GenerateFace(Faces::RIGHT, nodePos, resolution);
							GenerateFace(Faces::TOP, nodePos,   resolution);

							for (size_t x = 0; x < 2; x++)
							{
								for (size_t y = 0; y < 2; y++)
								{
									for (size_t z = 0; z < 2; z++)
									{

									}
								}
							}


							continue;
						}
						
						TraverseSVONode(innerNode->children[x][y][z], newChildResolution, newNodePosition, originPos, lodDistance);
						//std::cout << "DRAWING LOD LEVEL: " << resolutionLODlevel << std::endl;
					}
					else
						TraverseSVONode(innerNode->children[x][y][z], newChildResolution, newNodePosition, originPos, lodDistance);
					
					


				}
			}
		}
	
}
void ChunkMesh::CheckGenerationOfFace(Faces dir, SVOLeafNode* currLeafnode, glm::vec3 nodePos)
{
	SVOInnerNode* pParentNode = currLeafnode->pParentNode;
	glm::vec3 checkPosVec{ nodePos };
	int resolution = 2;
	int xID{};
	int yID{};
	int zID{};
	int currentID{};



	int IDToMatch{};
	bool isEvenNeeded{ false };
	xID = (int)checkPosVec.x % resolution;
	yID = (int)checkPosVec.y % resolution;
	zID = (int)checkPosVec.z % resolution;

	int xLocallookupID{xID};
	int yLocallookupID{yID};
	int zLocallookupID{zID};



	int ParentLookupID3D[3]{};
	int CheckPos;
	
	switch (dir)
	{
	case Faces::TOP:
		checkPosVec.y = checkPosVec.y + 1;
		CheckPos = checkPosVec.y;
		currentID = yID;
		yLocallookupID = currentID + 1;
		isEvenNeeded = true;

		break;
	case Faces::BOT:
		checkPosVec.y = checkPosVec.y - 1;
		CheckPos = checkPosVec.y;

		currentID = yID;
		yLocallookupID = currentID - 1;

		isEvenNeeded = false;
		break;
	case Faces::LEFT:
		checkPosVec.x = checkPosVec.x - 1;
		CheckPos = checkPosVec.x;

		currentID = xID;
		xLocallookupID = currentID - 1;
		isEvenNeeded = false;

		break;
	case Faces::RIGHT:
		checkPosVec.x = checkPosVec.x + 1;
		CheckPos = checkPosVec.x;

		currentID = xID;
		xLocallookupID = currentID + 1;
		isEvenNeeded = true;

		break;
	case Faces::FRONT:
		checkPosVec.z = checkPosVec.z + 1;
		CheckPos = checkPosVec.z;

		currentID = zID;
		zLocallookupID = currentID + 1;


		isEvenNeeded = true;
		break;
	case Faces::BACK:
		checkPosVec.z = checkPosVec.z - 1;
		CheckPos = checkPosVec.z;

		currentID = zID;
		zLocallookupID = currentID - 1;


		isEvenNeeded = false;

		break;
	default:
		break;
	}
	std::vector<int> debug;
	bool isEven = currentID % 2 == 0;
	if (isEven == (int)isEvenNeeded) { //CHECK IF WE CAN CHECK IN SAME OCTREE NODE
		//Select lookup leafnode from same parent;
		SVOBaseNode* basep = pParentNode->children[xLocallookupID][yLocallookupID][zLocallookupID];
		SVOLeafNode* leaf = static_cast<SVOLeafNode*>(basep);
		//CHECK IF VISIBLE OR INVISIBLE
		if (leaf->blockID != BlockTypes::AIR) {
			//BLOCKED DONT RENDER FACE
			return;
		}
		else
			GenerateFace(dir, nodePos);
	}
	else //Not IN the same octree
	{
		resolution = CHUNKSIZE_X;
		bool isInResolutionRange = false;
		int CurrentAxisLookupPos = CheckPos; //Current axis check pos
		//Is current Axis In Range Full resolution
		if ((CurrentAxisLookupPos) >= (resolution) || (CurrentAxisLookupPos) <= (-1)) {
			isInResolutionRange = false;
		}
		else
			isInResolutionRange = true;

		
		//Is in range
		if (isInResolutionRange) {
			//Scale other Axises ID with current Resolution
			SVOBaseNode* pNewBase = m_StarterNode;
			resolution = CHUNKSIZE_X;

			//Save local octree sector position
			int localSectorOffset[3]{0,0,0};
			do 
			{
				
				//If it is and end node stop here all data is further is sparse
				if (pNewBase->m_IsEndNode)
					return;


				for (size_t i = 0; i < 3; i++)
				{
					//Get axis position
					int position = (int)checkPosVec[i];

					//Check in which ID postion ios located
					if (position < (resolution / 2) + localSectorOffset[i]) {
						ParentLookupID3D[i] = 0;
					}
					else
						ParentLookupID3D[i] = 1;

					//Save current sector offset
					localSectorOffset[i] += (resolution / 2)  * ParentLookupID3D[i];
				}
				//Get octree child
				pNewBase = (static_cast<SVOInnerNode*>(pNewBase)->children[ParentLookupID3D[0]]
					[ParentLookupID3D[1]]
					[ParentLookupID3D[2]]);
				resolution *= 0.5f;
				//Untill we have found a leafnode
			} while (!(dynamic_cast<SVOLeafNode*>(pNewBase)));
			SVOLeafNode* leafNode = (static_cast<SVOLeafNode*>(pNewBase));

			if (leafNode->blockID != BlockTypes::AIR) {
				//BLOCKED DONT RENDER FACE
				return;
			}
			else {
				//BLOCK
				GenerateFace(dir, nodePos);
			}

			return;
		}

		//Draw faces at the end of our SVO resolution
		switch (dir)
		{
		case Faces::TOP:
			GenerateFace(dir, glm::vec3{ checkPosVec.x, CHUNKSIZE_Y - 1, checkPosVec.z });
			break;
		case Faces::BOT:
			GenerateFace(dir, glm::vec3{ checkPosVec.x, 0, checkPosVec.z });
			break;
		case Faces::LEFT:
			GenerateFace(dir, glm::vec3{ 0, checkPosVec.y, checkPosVec.z });
			break;
		case Faces::RIGHT:
			GenerateFace(dir, glm::vec3{ CHUNKSIZE_X - 1, checkPosVec.y, checkPosVec.z });
			break;
		case Faces::FRONT:
			GenerateFace(dir, glm::vec3{ checkPosVec.x, checkPosVec.y, CHUNKSIZE_X - 1 });
			break;
		case Faces::BACK:
			GenerateFace(dir, glm::vec3{ checkPosVec.x, checkPosVec.y,  0 });
			break;
		default:
			break;
		}
		return;
	}
}

void ChunkMesh::HasMultipleBlocks(SVOBaseNode* node, std::pair<bool, bool>& output)
{
	bool hasAir = false;
	bool hasBlock = false;

	if (node->m_IsEndNode)
		return;

	if (!dynamic_cast<SVOLeafNode*>(node)) {
		for (size_t x = 0; x < 2; x++)
		{
			for (size_t y = 0; y < 2; y++)
			{
				for (size_t z = 0; z < 2; z++)
				{
					SVOInnerNode* childNode = static_cast<SVOInnerNode*>(node);
					SVOBaseNode* pBaseNode = childNode->children[x][y][z];
					HasMultipleBlocks(pBaseNode, output);
				}

				
			}
		}
	}
	else {
		SVOLeafNode* leafnode = static_cast<SVOLeafNode*>(node);
		if (leafnode->blockID == AIR)
			output.first = true;
		else
			output.second = true;
	}
}



////void CheckGenerationOfFace1(Faces dir, SVOLeafNode* currLeafnode)
//{
//	SVOInnerNode* pParentNode = currLeafnode->pParentNode;
//	glm::vec3 checkPos{ currLeafnode->data};
//	int xID, yID, zID;
//	int resolution = 2;
//	xID = (int)checkPos.x % resolution;
//	yID = (int)checkPos.y % resolution;
//	zID = (int)checkPos.z % resolution;
//
//
//	switch (dir)
//	{
//	case Faces::TOP:
//			
//
//
//		break;
//	case Faces::BOT: 
//		checkPos.y -= 1;
//		if (yID == 1) { //Even
//			SVOBaseNode* basep = pParentNode->children[xID][yID - 1][zID];
//			SVOLeafNode* leaf = static_cast<SVOLeafNode*>(basep);
//			if (leaf->blockID == 1) {
//				return;
//			}
//			else
//				GenerateFace(Faces::BOT, currLeafnode->data);
//		}
//		else //Uneven go one layer above
//		{
//			SVOInnerNode* newParentNode = pParentNode->pParentNode;
//			do
//			{
//				resolution *= 2;
//				int newyID;
//				int newYpos = checkPos.y - 1;
//				newyID = (newYpos % (resolution)) >= (resolution * 0.5f) ? 0 : 1;
//				if (newyID == 0) {
//					SVOBaseNode* pNewBase = newParentNode->children[xID][newyID][zID];
//					while (!dynamic_cast<SVOLeafNode*>(pNewBase))
//					{
//						pNewBase = (static_cast<SVOInnerNode*>(pNewBase))->children[xID][newyID][zID];
//					}
//					SVOLeafNode* leafNode = (static_cast<SVOLeafNode*>(pNewBase));
//					if (leafNode->blockID == 1) {
//						//AIR
//						return;
//					}
//					else {
//						//BLOCK
//						GenerateFace(Faces::BOT, currLeafnode->data);
//					}
//
//					return;
//				}
//				else { //Go on Level Up
//					newParentNode = newParentNode->pParentNode;
//
//					if (checkPos.x == 0 && checkPos.z == 0) {
//						std::cout << checkPos.x << "|" << checkPos.z << "|" << resolution << std::endl;
//
//					}
//
//				}
//			} while (newParentNode != nullptr);
//
//
//			GenerateFace(Faces::BOT, glm::vec3{ checkPos.x, 0, checkPos.z });
//			return;
//		}
//		break;
//	case Faces::LEFT:
//		checkPos.x -= 1;
//		break;
//	case Faces::RIGHT:
//		checkPos.x += 1;
//		break;
//	case Faces::FRONT:
//		checkPos.z += 1;
//		break;
//	case Faces::BACK:
//		checkPos.z -= 1;
//		break;
//	default:
//		break;
//	}
//}

std::vector<glm::vec3>& ChunkMesh::GetVertices()
{
	return m_Vertices;
}

std::vector<int>& ChunkMesh::GetIndices()
{
	return m_Indices;
}



void ChunkMesh::FillSVO()
{

	//Fill starter node
	FillSVONode(m_StarterNode, MAX_DEPTH, 0, 0, 0, CHUNKSIZE_X);
	std::cout << "Amount of block filled: " << m_blockdetected << std::endl;
	m_blockdetected = 0;

}

void ChunkMesh::FillSVONode(SVOBaseNode* childToFill, int depth, int xPos, int yPos, int zPos, int resolution)
{
	
	if (resolution <= 1)
	{
		auto leafNode = static_cast<SVOLeafNode*>(childToFill);
		glm::vec3 position = glm::vec3{ xPos, yPos, zPos };

		//Generate Block
		BlockTypes terrainID = GetTerrainData(position);
		leafNode->blockID = terrainID;
		if (terrainID == AIR)
		{ //IS EMPTY

		}
		else {
			++m_blockdetected;
		}
	}
	else
	{
		int newResolution = resolution / 2;
		
#ifdef MAKE_SPARSE
		int localNodeID = resolution - (yPos % resolution);
		int topYpos = yPos + localNodeID;
		if (topYpos < (CHUNKSIZE_Y_MIN_TERRAIN - 1)) {
			childToFill->m_IsEndNode = true;
			return;
		}
#endif // MAKE_SPARSE

		SVOInnerNode* innerNode = static_cast<SVOInnerNode*>(childToFill);

		for (size_t x = 0; x < 2; x++)
		{
			for (size_t y = 0; y < 2; y++)
			{
				for (size_t z = 0; z < 2; z++)
				{
					if (newResolution <= 1) {
						//Fill in with leafnodes these are the end nodes
						SVOLeafNode* newNode = new SVOLeafNode();
						newNode->pParentNode = innerNode;
						//newNode->blockID = (rand() % 2 == 0 ? 0 : 1);
						innerNode->children[x][y][z] = newNode;

					}
					else {
						//Fill in with innernode we are not at the end
						SVOInnerNode* newNode = new SVOInnerNode();
						newNode->pParentNode = innerNode;
						innerNode->children[x][y][z] = newNode;
					}
				}
			}
		}
		
	
		FillSVONode(innerNode->children[0][0][0], depth - 1, xPos, yPos, zPos, newResolution);
		FillSVONode(innerNode->children[1][0][0], depth - 1, xPos + newResolution, yPos, zPos, newResolution);
		FillSVONode(innerNode->children[0][0][1], depth - 1, xPos, yPos, zPos + newResolution, newResolution);
		FillSVONode(innerNode->children[1][0][1], depth - 1, xPos + newResolution, yPos, zPos + newResolution, newResolution);

		FillSVONode(innerNode->children[0][1][0], depth - 1, xPos, yPos + newResolution, zPos, newResolution);
		FillSVONode(innerNode->children[1][1][0], depth - 1, xPos + newResolution, yPos + newResolution, zPos, newResolution);
		FillSVONode(innerNode->children[0][1][1], depth - 1, xPos, yPos + newResolution, zPos + newResolution, newResolution);
		FillSVONode(innerNode->children[1][1][1], depth - 1, xPos + newResolution, yPos + newResolution, zPos + newResolution, newResolution);


	}
	return;
}

BlockTypes ChunkMesh::GetTerrainData(glm::vec3 position)
{
	float value = static_cast<float>((perlin.octave2D_01((position.x) / 64.f, (position.z) / 64.f, 8)));
	float value2 = static_cast<float>((perlin.octave2D_01((position.x) / 128.f, (position.z) / 128.f, 8)));

	float totalValue = static_cast<float>((value * value2));
	//-0) / (1 - 0));
	int height = static_cast<int>(std::lerp(CHUNKSIZE_Y_MIN_TERRAIN, CHUNKSIZE_Y_MAX_TERRAIN, totalValue)) + 1;

	if (position.y >= height)
		return AIR;
	else
		return BlockTypes::BLOCK;

}
void ChunkMesh::GenerateFace(Faces dir, glm::vec3 position, int scale)
{
	const std::array<float, 12>* blockFace;
	std::vector<int> indices;
	std::vector<glm::vec3> vertices;

	switch (dir)
	{
	case Faces::TOP:
		blockFace = &topFace;
		break;
	case Faces::BOT:
		blockFace = &bottomFace;
		break;
	case Faces::LEFT:
		blockFace = &leftFace;
		break;
	case Faces::RIGHT:
		blockFace = &rightFace;
		break;
	case Faces::FRONT:
		blockFace = &frontFace;
		break;
	case Faces::BACK:
		blockFace = &backFace;
		break;
	default:
		break;
	}

	for (size_t i = 0; i < blockFace->size(); i += 3)
	{
		glm::vec3 Vertex1Pos = glm::vec3{ position.x + ((*blockFace)[i] * scale), position.y + ((*blockFace)[i + 1] * scale),
			position.z + ((*blockFace)[i + 2]* scale) };
		vertices.push_back(Vertex1Pos);
	}

	indices = { m_IndicesIndex, m_IndicesIndex + 1, m_IndicesIndex + 2, m_IndicesIndex + 2, m_IndicesIndex + 3, m_IndicesIndex };
	m_IndicesIndex += 4;
	m_ThreadIndices.insert(m_ThreadIndices.end(), indices.begin(), indices.end());
	m_ThreadVertices.insert(m_ThreadVertices.end(), vertices.begin(), vertices.end());

	m_Facedetected++;
}


