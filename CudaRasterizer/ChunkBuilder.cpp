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
		//GenerateFace(Faces::BACK,	leafNode->data);
		//GenerateFace(Faces::FRONT,	leafNode->data);
		//GenerateFace(Faces::LEFT,	leafNode->data);
		//GenerateFace(Faces::RIGHT,	leafNode->data);
		//GenerateFace(Faces::BOT,	leafNode->data);
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
	glm::vec3 checkPos{ currLeafnode->data};
	int xID, yID, zID;
	int resolution = 2;
	xID = (int)checkPos.x % resolution;
	yID = (int)checkPos.y % resolution;
	zID = (int)checkPos.z % resolution;


	switch (dir)
	{
	case Faces::TOP:
		if (yID == 0) { //Even
			SVOBaseNode* basep = pParentNode->children[xID][yID + 1][zID];
			SVOLeafNode* leaf =	static_cast<SVOLeafNode*>(basep);
			if (leaf->blockID == 1) {
				return;
			}
			else
				GenerateFace(Faces::TOP, currLeafnode->data);
		}
		else //Uneven go one layer above
		{
			SVOInnerNode* newParentNode = pParentNode->pParentNode;
			do 
			{
				resolution *= 2;
				int newyID;
				int newYpos = checkPos.y + 1;
				newyID = (newYpos % (resolution)) >= (resolution *0.5f) ? 1 : 0;
				if (newyID == 1) {
					SVOBaseNode* pNewBase = newParentNode->children[xID][newyID][zID];
					while (!dynamic_cast<SVOLeafNode*>(pNewBase))
					{
						pNewBase = (static_cast<SVOInnerNode*>(pNewBase))->children[xID][newyID][zID];
					}
					SVOLeafNode* leafNode = (static_cast<SVOLeafNode*>(pNewBase));
					if (leafNode->blockID == 1) {
						//AIR
						return;
					}
					else {
						//BLOCK
						GenerateFace(Faces::TOP, currLeafnode->data);
					}

					return;
				}
				else { //Go on Level Up
					newParentNode = newParentNode->pParentNode;

					if (checkPos.x == 0 && checkPos.z == 0) {
						std::cout << checkPos.x << "|" << checkPos.z << "|" << resolution << std::endl;

					}
				
				}
			} while (newParentNode != nullptr);

			
			GenerateFace(Faces::TOP, glm::vec3{checkPos.x, CHUNKSIZE_Y - 1, checkPos.z});
			return;
		}	


		break;
	case Faces::BOT: 
		checkPos.y -= 1;
		break;
	case Faces::LEFT:
		checkPos.x -= 1;
		break;
	case Faces::RIGHT:
		checkPos.x += 1;
		break;
	case Faces::FRONT:
		checkPos.z += 1;
		break;
	case Faces::BACK:
		checkPos.z -= 1;
		break;
	default:
		break;
	}
}

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
						newNode->blockID = 1;
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


