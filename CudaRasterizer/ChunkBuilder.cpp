#include "ChunkBuilder.h"
#include <array>
#include <iostream>

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

ChunkMesh::ChunkMesh()
{
	m_StarterNode = new SVOInnerNode();
	m_StarterNode->pParentNode = nullptr;
	FillSVO();
}

ChunkMesh::~ChunkMesh()
{
	if (m_StarterNode != nullptr) {
		delete m_StarterNode;
	}
}

void ChunkMesh::TraverseSVO()
{
	TraverseSVONode(m_StarterNode, MAX_DEPTH);
	std::cout << "Amount of block filled: " << m_blockdetected << std::endl;

}
void ChunkMesh::TraverseSVONode(SVOBaseNode* pNode, int depth)
{
	if (pNode == nullptr)
		return;

	if (dynamic_cast<SVOLeafNode*>(pNode)) {
		//FILL IN vertex information
		auto leafNode = static_cast<SVOLeafNode*>(pNode);
		if (leafNode->blockID != AIR) {
			++m_blockdetected;
			CheckGenerationOfFace(Faces::TOP, leafNode);
			CheckGenerationOfFace(Faces::BOT, leafNode);
			CheckGenerationOfFace(Faces::FRONT, leafNode);
			CheckGenerationOfFace(Faces::BACK, leafNode);
			CheckGenerationOfFace(Faces::LEFT, leafNode);
			CheckGenerationOfFace(Faces::RIGHT, leafNode);
		}
		return; //THIS NODE IS AN END NODE
	}
	SVOInnerNode* innerNode = static_cast<SVOInnerNode*>(pNode);
	for (size_t x = 0; x < 2; x++)
	{
		for (size_t y = 0; y < 2; y++)
		{
			for (size_t z = 0; z < 2; z++)
			{
				TraverseSVONode(innerNode->children[x][y][z], depth - 1);

			}
		}
	}

}
void ChunkMesh::CheckGenerationOfFace(Faces dir, SVOLeafNode* currLeafnode)
{
	SVOInnerNode* pParentNode = currLeafnode->pParentNode;
	glm::vec3 checkPosVec{ currLeafnode->data };
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
			GenerateFace(dir, currLeafnode->data);
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
				GenerateFace(dir, currLeafnode->data);
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
	return BlockTypes::BLOCK;
}
void ChunkMesh::GenerateFace(Faces dir, glm::vec3 position)
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
		glm::vec3 Vertex1Pos = glm::vec3{ position.x + (*blockFace)[i], position.y + (*blockFace)[i + 1],position.z + (*blockFace)[i + 2] };
		vertices.push_back(Vertex1Pos);
	}

	indices = { m_IndicesIndex, m_IndicesIndex + 1, m_IndicesIndex + 2, m_IndicesIndex + 2, m_IndicesIndex + 3, m_IndicesIndex };
	m_IndicesIndex += 4;
	m_Indices.insert(m_Indices.end(), indices.begin(), indices.end());
	m_Vertices.insert(m_Vertices.end(), vertices.begin(), vertices.end());

	m_Facedetected++;
}


