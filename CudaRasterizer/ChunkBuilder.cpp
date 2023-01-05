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

}
void ChunkMesh::TraverseSVONode(SVOBaseNode* pNode, int depth)
{
	if (pNode == nullptr)
		return;

	if (dynamic_cast<SVOLeafNode*>(pNode)) {
		//FILL IN vertex information
		auto leafNode = static_cast<SVOLeafNode*>(pNode);
		++m_blockdetected;

		CheckGenerationOfFace(Faces::TOP, leafNode);
		CheckGenerationOfFace(Faces::BOT, leafNode);
		CheckGenerationOfFace(Faces::BACK,	leafNode);
		CheckGenerationOfFace(Faces::FRONT,	leafNode);
		CheckGenerationOfFace(Faces::LEFT,	leafNode);
		CheckGenerationOfFace(Faces::RIGHT,	leafNode);
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

	int xlookupID{xID};
	int ylookupID{yID};
	int zlookupID{zID};

	int CheckPos{};
	switch (dir)
	{
	case Faces::TOP:
		CheckPos = checkPosVec.y + 1;
		currentID = yID;
		ylookupID = currentID + 1;
		IDToMatch = 1;
		isEvenNeeded = true;
		yID = 0;
		break;
	case Faces::BOT:
		CheckPos = checkPosVec.y - 1;
		currentID = yID;
		ylookupID = currentID - 1;
		IDToMatch = 0;
		isEvenNeeded = false;
		yID = 1;
		break;
	case Faces::LEFT:
		CheckPos = checkPosVec.x - 1;
		currentID = xID;
		xlookupID = currentID - 1;
		isEvenNeeded = false;
		xID = 1;

		break;
	case Faces::RIGHT:
		CheckPos = checkPosVec.x + 1;
		currentID = xID;
		xlookupID = currentID + 1;
		isEvenNeeded = true;
		xID = 0;
		break;
	case Faces::FRONT:
		CheckPos = checkPosVec.z + 1;
		currentID = zID;
		zlookupID = currentID + 1;
		zID = 0;
		isEvenNeeded = true;
		break;
	case Faces::BACK:
		CheckPos = checkPosVec.z - 1;
		currentID = zID;
		zlookupID = currentID - 1;
		zID = 1;
		isEvenNeeded = false;

		break;
	default:
		break;
	}
	std::vector<int> debug;
	bool isEven = currentID % 2 == 0;
	if (isEven == (int)isEvenNeeded) { //Even
		//Select node in same parent node
		SVOBaseNode* basep = pParentNode->children[xlookupID][ylookupID][zlookupID];
		SVOLeafNode* leaf = static_cast<SVOLeafNode*>(basep);
		if (leaf->blockID == BlockTypes::AIR) {
			//AIR
			//GenerateFace(dir, currLeafnode->data);
			return;
		}
		else
			GenerateFace(dir, currLeafnode->data);
	}
	else //Uneven go one layer above
	{
		SVOInnerNode* newParentNode = pParentNode->pParentNode;
		resolution = 4;
		do
		{

			int newLocalID;

			int newLookupPos = CheckPos;
			/*if (newLookupPos >= 15) {
				std::cout << checkPosVec.x << "|" << checkPosVec.z << "|" << resolution << std::endl;
			}*/
			if(isEvenNeeded)
				newLocalID = (newLookupPos) >= (resolution - 1) ? 1 : 0;
			else
				newLocalID = (newLookupPos) <= (0 - 1) ? 1 : 0;

			debug.push_back(newLookupPos);
			debug.push_back(resolution);
			resolution *= 2;

			if (newLocalID == 0) {
				//This has to be inverted when we areg going the opposite way look in 1 for child but go down to 0 for TOP
				SVOBaseNode* pNewBase;
				if (isEvenNeeded)
					pNewBase = newParentNode->children[xID][yID][zID];
				else
					pNewBase = newParentNode->children[xID][yID][zID];

				while (!dynamic_cast<SVOLeafNode*>(pNewBase))
				{
					if (isEvenNeeded)
						pNewBase = (static_cast<SVOInnerNode*>(pNewBase))->children[xID][yID][zID];
					else
						pNewBase = (static_cast<SVOInnerNode*>(pNewBase))->children[xID][yID][zID];

				}
				SVOLeafNode* leafNode = (static_cast<SVOLeafNode*>(pNewBase));
				if (leafNode->blockID == BlockTypes::AIR) {
					//AIR
					//GenerateFace(dir, currLeafnode->data);
					return;
				}
				else {
					//BLOCK
					GenerateFace(dir, currLeafnode->data);
				}

				return;
			}
			else { //Go on Level Up
				newParentNode = newParentNode->pParentNode;
			}
		} while (newParentNode != nullptr);


		//Draw end of SVO 
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
		/*if (checkPosVec.y == 0 && checkPosVec.x == 0) {
			debug;
			std::cout << checkPosVec.x << "|" << checkPosVec.z << "|" << resolution << std::endl;

		}*/
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

}

void ChunkMesh::FillSVONode(SVOBaseNode* childToFill, int depth, int xPos, int yPos, int zPos, int resolution)
{
	
	if (resolution <= 1)
	{
		auto leafNode = static_cast<SVOLeafNode*>(childToFill);
		//Generate Block
		bool terrainID = GetTerrainData();
		if (terrainID == false)
		{ //IS EMPTY
		}
		glm::vec3 position = glm::vec3{ xPos, yPos, zPos };
		leafNode->data = position;
		/*if (xPos < 2) {
			leafNode->blockID = 0;
		}
		else*/
		leafNode->blockID = BlockTypes::AIR;


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

bool ChunkMesh::GetTerrainData()
{
	return true;
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


