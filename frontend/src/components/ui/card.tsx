/**
 * Minimalistic Card Component
 * Clean white cards with subtle borders and sharp corners
 */

import React from 'react'

export interface CardProps extends React.HTMLAttributes<HTMLDivElement> {
  header?: React.ReactNode;
  title?: string;
  description?: string;
}

export const Card = React.forwardRef<HTMLDivElement, CardProps>(
  ({ className = '', children, header, title, description, ...props }, ref) => {
    return (
      <div
        ref={ref}
        className={`bg-white border border-gray-200 rounded-sm ${className}`}
        {...props}
      >
        {(header || title || description) && (
          <div className="px-5 py-4 border-b border-gray-100">
            {header || (
              <>
                {title && (
                  <h2 className="text-xs font-medium text-gray-900 uppercase tracking-wide">
                    {title}
                  </h2>
                )}
                {description && (
                  <p className="text-xs font-light text-gray-500 mt-1">
                    {description}
                  </p>
                )}
              </>
            )}
          </div>
        )}
        <div className="p-5">
          {children}
        </div>
      </div>
    )
  }
)

Card.displayName = 'Card'

export interface CardHeaderProps extends React.HTMLAttributes<HTMLDivElement> {}

export const CardHeader = React.forwardRef<HTMLDivElement, CardHeaderProps>(
  ({ className = '', children, ...props }, ref) => {
    return (
      <div
        ref={ref}
        className={`px-5 py-4 border-b border-gray-100 ${className}`}
        {...props}
      >
        {children}
      </div>
    )
  }
)

CardHeader.displayName = 'CardHeader'

export interface CardContentProps extends React.HTMLAttributes<HTMLDivElement> {}

export const CardContent = React.forwardRef<HTMLDivElement, CardContentProps>(
  ({ className = '', children, ...props }, ref) => {
    return (
      <div
        ref={ref}
        className={`p-5 ${className}`}
        {...props}
      >
        {children}
      </div>
    )
  }
)

CardContent.displayName = 'CardContent'

export interface CardTitleProps extends React.HTMLAttributes<HTMLHeadingElement> {}

export const CardTitle = React.forwardRef<HTMLHeadingElement, CardTitleProps>(
  ({ className = '', children, ...props }, ref) => {
    return (
      <h3
        ref={ref}
        className={`text-xs font-medium text-gray-900 uppercase tracking-wide ${className}`}
        {...props}
      >
        {children}
      </h3>
    )
  }
)

CardTitle.displayName = 'CardTitle'

export interface CardDescriptionProps extends React.HTMLAttributes<HTMLParagraphElement> {}

export const CardDescription = React.forwardRef<HTMLParagraphElement, CardDescriptionProps>(
  ({ className = '', children, ...props }, ref) => {
    return (
      <p
        ref={ref}
        className={`text-xs font-light text-gray-500 mt-1 ${className}`}
        {...props}
      >
        {children}
      </p>
    )
  }
)

CardDescription.displayName = 'CardDescription'
