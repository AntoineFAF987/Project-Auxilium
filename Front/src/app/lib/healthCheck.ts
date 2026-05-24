// Utilitaire pour vérifier la santé du serveur backend

export async function checkBackendHealth(): Promise<{
  isHealthy: boolean;
  message: string;
  responseTime?: number;
}> {
  const startTime = Date.now();
  
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 5000);
    
    const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/status`, {
      signal: controller.signal,
      cache: "no-store",
    });
    
    clearTimeout(timeoutId);
    const responseTime = Date.now() - startTime;
    
    if (response.ok) {
      return {
        isHealthy: true,
        message: `Serveur accessible (${responseTime}ms)`,
        responseTime,
      };
    } else {
      return {
        isHealthy: false,
        message: `Serveur répond avec erreur ${response.status}`,
        responseTime,
      };
    }
  } catch (err: any) {
    const responseTime = Date.now() - startTime;
    
    if (err.name === 'AbortError') {
      return {
        isHealthy: false,
        message: "Timeout - le serveur ne répond pas dans les 5 secondes",
        responseTime,
      };
    }
    
    return {
      isHealthy: false,
      message: `Impossible de contacter le serveur: ${err.message || err}`,
      responseTime,
    };
  }
}
