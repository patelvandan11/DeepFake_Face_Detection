'use client';

import { GoogleOAuthProvider } from '@react-oauth/google';

export function Providers({ children }: { children: React.ReactNode }) {
  const googleClientId = process.env.NEXT_PUBLIC_GOOGLE_CLIENT_ID;

  console.log("Client ID being used:", googleClientId);
  
  if (!googleClientId) {
    throw new Error("Missing NEXT_PUBLIC_GOOGLE_CLIENT_ID environment variable");
  }

  return (
    <GoogleOAuthProvider clientId={googleClientId}>
      {children}
    </GoogleOAuthProvider>
  );
}
