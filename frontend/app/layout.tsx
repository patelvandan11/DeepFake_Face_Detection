import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'
import Navbar from './components/Navbar'
import Footer from './components/Footer'
import BlackFireCursor from './components/BlackFireCursor'
import { Providers } from './providers';

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'Unmasked',
  description: 'Detect deepfake images with AI-powered technology',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {

  return (
    <html lang="en" className="dark">
      <body className={`${inter.className} bg-dark-100 text-gray-100 min-h-screen`}>
        {/* The Providers component should wrap all the content */}
        <Providers>
          <Navbar />
          <BlackFireCursor />
          <main className="pt-20 pb-10">
            {children} {/* children is now rendered only ONCE */}
          </main>
          <Footer />
        </Providers>
      </body>
    </html>
  )
}