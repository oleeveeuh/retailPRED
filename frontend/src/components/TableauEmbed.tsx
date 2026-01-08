/**
 * Tableau Embed Component
 *
 * Properly embeds Tableau Public visualizations using the Tableau JavaScript API
 */

import { FC, useEffect, useRef } from 'react';

interface TableauEmbedProps {
  url: string;
  height?: string | number;
}

export const TableauEmbed: FC<TableauEmbedProps> = ({ url, height = 600 }) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const scriptLoaded = useRef(false);

  useEffect(() => {
    if (!containerRef.current || scriptLoaded.current) return;

    // Parse the Tableau URL to extract workbook and sheet info
    // Format: https://public.tableau.com/views/Book1_17676501972860/Sheet1
    const urlParts = url.split('public.tableau.com/');
    if (urlParts.length < 2) return;

    const pathPart = urlParts[1].split('?')[0]; // Remove query params
    const workbookAndSheet = pathPart.replace('views/', '');
    const [workbook, sheet] = workbookAndSheet.split('/');

    // Generate a unique viz ID
    const vizId = `viz${Date.now()}`;

    // Create the Tableau embed HTML
    const embedHTML = `
      <div class='tableauPlaceholder' id='${vizId}' style='position: relative; width: 100%; height: 100%; min-height: ${typeof height === 'number' ? height + 'px' : height};'>
        <noscript>
          <a href='#'>
            <img alt='Tableau Visualization' src='https://public.tableau.com/static/images/${workbook.charAt(0)}/${workbook}/${sheet}/1_rss.png' style='border: none' />
          </a>
        </noscript>
        <object class='tableauViz' style='display:none; width:100%; height:100%;'>
          <param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' />
          <param name='embed_code_version' value='3' />
          <param name='site_root' value='' />
          <param name='name' value='${workbook}/${sheet}' />
          <param name='tabs' value='no' />
          <param name='toolbar' value='yes' />
          <param name='static_image' value='https://public.tableau.com/static/images/${workbook.charAt(0)}/${workbook}/${sheet}/1.png' />
          <param name='animate_transition' value='yes' />
          <param name='display_static_image' value='yes' />
          <param name='display_spinner' value='yes' />
          <param name='display_overlay' value='yes' />
          <param name='display_count' value='yes' />
          <param name='language' value='en-US' />
        </object>
      </div>
    `;

    // Insert the HTML
    containerRef.current.innerHTML = embedHTML;

    // Load the Tableau JavaScript API
    const script = document.createElement('script');
    script.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';
    script.onload = () => {
      scriptLoaded.current = true;
    };
    containerRef.current.appendChild(script);

    return () => {
      // Cleanup
      if (containerRef.current) {
        containerRef.current.innerHTML = '';
      }
      scriptLoaded.current = false;
    };
  }, [url, height]);

  return (
    <div
      ref={containerRef}
      className="tableau-embed-container"
      style={{ width: '100%', height: '100%', minHeight: typeof height === 'number' ? height + 'px' : height }}
    />
  );
};
