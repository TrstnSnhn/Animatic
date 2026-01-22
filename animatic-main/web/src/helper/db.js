import { openDB } from "idb";

const DB_NAME = "MyGLBFilesDB";
const STORE_NAME = "glbFiles";

export async function getDb() {
  return openDB(DB_NAME, 1, {
    upgrade(db) {
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME, { keyPath: "id", autoIncrement: true });
      }
    },
  });
}

export async function saveFile(fileData) {
  const db = await getDb();
  await db.add(STORE_NAME, fileData); // fileData = { fileData, filename, image }
}

export async function getAllFiles() {
  const db = await getDb();
  return db.getAll(STORE_NAME);
}

export async function deleteFile(id) {
  const db = await getDb();
  await db.delete(STORE_NAME, id);
}