#!/usr/bin/env node
/**
 * LexiAI Frontend Deployment Script
 *
 * Creates production-ready HTML files that use minified JavaScript
 * - Replaces all .js references with .min.js
 * - Copies minified JS files
 * - Preserves original development files
 * - Creates production build in dist/ directory
 */

const fs = require('fs');
const path = require('path');

// Configuration
const CONFIG = {
  frontendDir: path.join(__dirname, '..'),
  distDir: path.join(__dirname, '../dist'),
  minifiedJsDir: path.join(__dirname, '../js/dist'),
  skipDirs: ['node_modules', 'scripts', 'dist', '.git'],
};

// Color output
const colors = {
  reset: '\x1b[0m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  red: '\x1b[31m',
};

function log(msg, color = 'reset') {
  console.log(`${colors[color]}${msg}${colors.reset}`);
}

// Copy directory recursively
function copyDir(src, dest, transform = null) {
  if (!fs.existsSync(dest)) {
    fs.mkdirSync(dest, { recursive: true });
  }

  const entries = fs.readdirSync(src, { withFileTypes: true });

  for (const entry of entries) {
    const srcPath = path.join(src, entry.name);
    const destPath = path.join(dest, entry.name);

    // Skip excluded directories
    if (entry.isDirectory() && CONFIG.skipDirs.includes(entry.name)) {
      continue;
    }

    if (entry.isDirectory()) {
      copyDir(srcPath, destPath, transform);
    } else {
      // Apply transformation if provided
      if (transform && (entry.name.endsWith('.html') || entry.name.endsWith('.htm'))) {
        const content = fs.readFileSync(srcPath, 'utf8');
        const transformed = transform(content, entry.name);
        fs.writeFileSync(destPath, transformed, 'utf8');
      } else {
        fs.copyFileSync(srcPath, destPath);
      }
    }
  }
}

// Transform HTML to use minified JS
function transformHTML(content, filename) {
  let transformed = content;
  let replacements = 0;

  // Replace all .js references with .min.js
  // Pattern: src="path/to/file.js" or src="/path/to/file.js"
  transformed = transformed.replace(
    /src=["']([^"']*?\/js\/([^"']*?)\.js)["']/gi,
    (match, fullPath, filename) => {
      // Skip if already minified
      if (filename.endsWith('.min')) {
        return match;
      }

      replacements++;
      // Replace with minified version in dist directory
      return `src="/frontend/js/dist/${filename}.min.js"`;
    }
  );

  return { content: transformed, replacements };
}

// Main deployment function
async function deploy() {
  log('\n🚀 LexiAI Frontend Deployment', 'yellow');
  log('═══════════════════════════════════\n', 'yellow');

  const startTime = Date.now();

  // Clean dist directory
  if (fs.existsSync(CONFIG.distDir)) {
    log('🧹 Cleaning previous build...', 'blue');
    fs.rmSync(CONFIG.distDir, { recursive: true, force: true });
  }

  // Create dist directory
  fs.mkdirSync(CONFIG.distDir, { recursive: true });

  // Track statistics
  const stats = {
    htmlFiles: 0,
    jsReplacements: 0,
    cssFiles: 0,
    otherFiles: 0,
  };

  // Transform and copy HTML files
  log('📄 Processing HTML files...', 'blue');
  const htmlTransform = (content, filename) => {
    const result = transformHTML(content, filename);
    stats.htmlFiles++;
    stats.jsReplacements += result.replacements;

    if (result.replacements > 0) {
      log(`  ✓ ${filename} (${result.replacements} JS references updated)`, 'green');
    }

    return result.content;
  };

  copyDir(CONFIG.frontendDir, CONFIG.distDir, htmlTransform);

  // Copy minified JS files
  log('\n📦 Copying minified JavaScript...', 'blue');
  const distJsDir = path.join(CONFIG.distDir, 'js/dist');
  if (!fs.existsSync(distJsDir)) {
    fs.mkdirSync(distJsDir, { recursive: true });
  }

  if (fs.existsSync(CONFIG.minifiedJsDir)) {
    const minFiles = fs.readdirSync(CONFIG.minifiedJsDir);
    for (const file of minFiles) {
      if (file.endsWith('.min.js') || file.endsWith('.min.js.map')) {
        fs.copyFileSync(
          path.join(CONFIG.minifiedJsDir, file),
          path.join(distJsDir, file)
        );
      }
    }
    log(`  ✓ Copied ${minFiles.filter(f => f.endsWith('.min.js')).length} minified files`, 'green');
  }

  // Calculate build time
  const buildTime = ((Date.now() - startTime) / 1000).toFixed(2);

  // Summary
  log('\n═══════════════════════════════════', 'yellow');
  log('📊 Deployment Summary', 'yellow');
  log('═══════════════════════════════════\n', 'yellow');
  log(`HTML files:       ${stats.htmlFiles}`, 'blue');
  log(`JS replacements:  ${stats.jsReplacements}`, 'green');
  log(`Build time:       ${buildTime}s`, 'blue');
  log(`Output directory: ${CONFIG.distDir}\n`, 'blue');

  log('✅ Deployment completed successfully!\n', 'green');
  log('📝 Next steps:', 'yellow');
  log('   1. Test the production build: open dist/index.html', 'blue');
  log('   2. Verify minified JS loads correctly', 'blue');
  log('   3. Deploy dist/ directory to production server\n', 'blue');
}

// Run deployment
deploy().catch(error => {
  log(`\n❌ Deployment failed: ${error.message}\n`, 'red');
  console.error(error);
  process.exit(1);
});
