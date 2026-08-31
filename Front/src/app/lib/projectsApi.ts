export type Project = {
  id: string;
  name: string;
  created_at: string;
  updated_at: string;
};

async function request(path: string, idToken: string, init: RequestInit = {}) {
  const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}${path}`, {
    ...init,
    headers: { Authorization: `Bearer ${idToken}`, ...(init.headers || {}) },
    cache: "no-store",
  });
  if (!response.ok) throw new Error(`HTTP ${response.status}: ${await response.text()}`);
  return response;
}

export async function fetchProjects(idToken: string): Promise<Project[]> {
  const response = await request("/projects", idToken);
  const data = await response.json();
  return Array.isArray(data?.items) ? data.items : [];
}

export async function createProject(name: string, idToken: string): Promise<Project> {
  return (await request("/projects", idToken, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name }) })).json();
}

export async function renameProject(projectId: string, name: string, idToken: string): Promise<Project> {
  return (await request(`/projects/${projectId}`, idToken, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name }) })).json();
}

export async function deleteProject(projectId: string, idToken: string): Promise<void> {
  await request(`/projects/${projectId}`, idToken, { method: "DELETE" });
}
