#![deny(missing_docs)]

/*!
This crate provides an implementation for calculating the positions of text
fragments to fit text withing a rectangle.

The `graphics` feature, which is on by default, allows the direct rendering of a fitted
text with the [`piston2d-graphics`](https://docs.rs/piston2d-graphics) crate.
*/

use std::{collections::HashMap, sync::Mutex};

#[cfg(feature = "graphics")]
use graphics::{
    character::CharacterCache, math::Matrix2d, text as draw_text, Graphics, ImageSize, Transformed,
};
use once_cell::sync::Lazy;
use rusttype::{Error, Font, GlyphId, Scale};
use vector2math::*;

/// A horizantal text justification
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Justification {
    /// Align on the left
    Left,
    /// Center align
    Centered,
    /// Align on the right
    Right,
}

/// Lines that have starting positions
///
/// `V` usually implements `Vector2`
pub type PositionedLines<V> = Vec<(V, String)>;

/// A way of resizing text in a rectangle
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Resize {
    /// Make the text no larger than its original font size,
    /// but still try to fit it in the rectangle
    NoLarger,
    /// Make the text as large as possible while still
    /// fitting in the rectangle
    Max,
    /// Do not resize the text
    None,
}

/// A format for some text
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct TextFormat {
    /// The font size
    pub font_size: u32,
    /// The horizantal justification
    pub just: Justification,
    /// The spacing between lines. This should usually be somewhere
    /// between `1.0` and `2.0`, but any scalar is valid
    pub line_spacing: f64,
    /// The number of spaces to indent the first line of a paragraph
    pub first_line_indent: usize,
    /// The number of spaces to indent all lines of a paragraph
    /// after the first
    pub lines_indent: usize,
    /// The color of the text
    pub color: Color,
    /// The resize strategy
    pub resize: Resize,
}

impl From<u32> for TextFormat {
    fn from(font_size: u32) -> Self {
        TextFormat::new(font_size)
    }
}

static DEFAULT_TEXT_FORMAT: Lazy<Mutex<TextFormat>> = Lazy::new(|| {
    Mutex::new(TextFormat {
        font_size: 30,
        just: Justification::Left,
        line_spacing: 1.0,
        first_line_indent: 0,
        lines_indent: 0,
        color: [0.0, 0.0, 0.0, 1.0],
        resize: Resize::NoLarger,
    })
});

impl Default for TextFormat {
    fn default() -> TextFormat {
        *DEFAULT_TEXT_FORMAT
            .lock()
            .expect("fit-text default TextFormat thread panicked")
    }
}

impl TextFormat {
    /// Create a default `TextFormat` with the given font size
    pub fn new(font_size: u32) -> TextFormat {
        TextFormat {
            font_size,
            ..Default::default()
        }
    }
    /// Use this `TextFormat` as the default
    pub fn set_as_default(&self) {
        *DEFAULT_TEXT_FORMAT
            .lock()
            .expect("fit-text default TextFormat thread panicked") = *self;
    }
    /// Align the `TextFormat` to the left
    pub fn left(mut self) -> Self {
        self.just = Justification::Left;
        self
    }
    /// Center-align the `TextFormat`
    pub fn centered(mut self) -> Self {
        self.just = Justification::Centered;
        self
    }
    /// Align the `TextFormat` to the right
    pub fn right(mut self) -> Self {
        self.just = Justification::Right;
        self
    }
    /// Set the font size
    pub fn font_size(mut self, font_size: u32) -> Self {
        self.font_size = font_size;
        self
    }
    /// Set the line spacing
    pub fn line_spacing(mut self, line_spacing: f64) -> Self {
        self.line_spacing = line_spacing;
        self
    }
    /// Set the indentation of the first line
    pub fn first_line_indent(mut self, first_line_indent: usize) -> Self {
        self.first_line_indent = first_line_indent;
        self
    }
    /// Set the indentation of all lines after the first
    pub fn lines_indent(mut self, lines_indent: usize) -> Self {
        self.lines_indent = lines_indent;
        self
    }
    /// Set the color
    pub fn color(mut self, color: Color) -> Self {
        self.color = color;
        self
    }
    /// Set the resize strategy
    pub fn resize(mut self, resize: Resize) -> Self {
        self.resize = resize;
        self
    }
    /// Change the font size depending on the the resize strategy
    ///
    /// The given max size is not used if the strategy is `Resize::None`
    pub fn resize_font(mut self, max_size: u32) -> Self {
        match self.resize {
            Resize::NoLarger => self.font_size = self.font_size.min(max_size),
            Resize::Max => self.font_size = max_size,
            Resize::None => (),
        }
        self
    }
}

/// Defines behavior of a cache of character widths.
///
/// In general, determining the width of a character glyphs with a given font size
/// is a non-trivial calculation. Caching a width calculation for each characters
/// and font size ensures that the calculation is only done once for each pair.
pub trait CharacterWidthCache {
    /// Get the width of a character at a font size
    fn char_width(&mut self, character: char, font_size: u32) -> f64;
    /// Get the width of a string at a font_size
    fn width(&mut self, text: &str, font_size: u32) -> f64 {
        text.chars()
            .map(|c| self.char_width(c, font_size))
            .fold(0.0, std::ops::Add::add)
    }
    /// Split a string into a list of lines of text with the given format where no line
    /// is wider than the given max width. Newlines (`\n`) in the string are respected
    fn format_lines<F>(&mut self, text: &str, max_width: f64, format: F) -> Vec<String>
    where
        F: Into<TextFormat>,
    {
        let format = format.into();
        let mut sized_lines = Vec::new();
        let mut first_line = false;
        // Iterate through lines
        for line in text.lines() {
            // Initialize a result line
            let mut sized_line = String::new();
            // Apply the indentation
            let indent = (0..if first_line {
                format.first_line_indent
            } else {
                format.lines_indent
            })
                .map(|_| ' ')
                .collect::<String>();
            sized_line.push_str(&indent);
            let mut curr_width = self.width(&indent, format.font_size);
            // Iterate through words
            for word in line.split_whitespace() {
                // Get the word's width
                let width = self.width(word, format.font_size);
                // If the word goes past the max width...
                if !(curr_width + width < max_width || curr_width == 0.0) {
                    // Pop off the trailing space
                    sized_line.pop();
                    // Push the result line onto the result list
                    sized_lines.push(sized_line);
                    // Init next line
                    first_line = false;
                    sized_line = String::new();
                    // Apply the indentation
                    let indent = (0..if first_line {
                        format.first_line_indent
                    } else {
                        format.lines_indent
                    })
                        .map(|_| ' ')
                        .collect::<String>();
                    sized_line.push_str(&indent);
                    curr_width = self.width(&indent, format.font_size);
                }
                // Push the word onto the result line
                sized_line.push_str(word);
                sized_line.push(' ');
                curr_width = curr_width + width + self.char_width(' ', format.font_size);
            }
            // Push the result line onto the result list
            sized_line.pop();
            sized_lines.push(sized_line);
            first_line = false;
        }
        sized_lines
    }
    /// Get the width of the widest line after performing
    /// the calculation of `CharacterWidthCache::format_lines`
    fn max_line_width<F>(&mut self, text: &str, max_width: f64, format: F) -> f64
    where
        F: Into<TextFormat>,
    {
        let format = format.into();
        let lines = self.format_lines(text, max_width, format);
        lines
            .into_iter()
            .map(|line| self.width(&line, format.font_size))
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0)
    }
    /// Calculate a set of positioned lines of text with the given format
    /// that fit within the given rectangle
    fn justify_text<R, F>(&mut self, text: &str, rect: R, format: F) -> PositionedLines<R::Vector>
    where
        R: Rectangle<Scalar = f64>,
        F: Into<TextFormat>,
    {
        let format = format.into();
        self.format_lines(text, rect.width(), format)
            .into_iter()
            .enumerate()
            .map(|(i, line)| {
                let y_offset = rect.top()
                    + format.font_size as f64
                    + i as f64 * format.font_size as f64 * format.line_spacing;
                use self::Justification::*;
                let line_width = self.width(&line, format.font_size);
                let x_offset = match format.just {
                    Left => rect.left(),
                    Centered => rect.center().x() - line_width / 2.0,
                    Right => rect.right() - line_width,
                };
                (R::Vector::new(x_offset, y_offset), line)
            })
            .collect()
    }
    /// Check if text with the given format fits within a rectangle's width
    fn text_fits_horizontal<R, F>(&mut self, text: &str, rect: R, format: F) -> bool
    where
        R: Rectangle<Scalar = f64>,
        F: Into<TextFormat>,
    {
        self.max_line_width(text, rect.width(), format) < rect.width()
    }
    /// Check if text with the given format fits within a rectangle's height
    fn text_fits_vertical<R, F>(&mut self, text: &str, rect: R, format: F) -> bool
    where
        R: Rectangle<Scalar = f64>,
        F: Into<TextFormat>,
    {
        let format = format.into();
        let lines = self.format_lines(text, rect.width(), format);
        if lines.is_empty() {
            return true;
        }
        let last_line_y = rect.top()
            + format.font_size as f64
            + (lines.len() - 1) as f64 * format.font_size as f64 * format.line_spacing;
        last_line_y < rect.bottom()
    }
    /// Check if text with the given format fits within a rectangle
    fn text_fits<R, F>(&mut self, text: &str, rect: R, format: F) -> bool
    where
        R: Rectangle<Scalar = f64>,
        F: Into<TextFormat>,
    {
        let format = format.into();
        self.text_fits_horizontal(text, rect, format) && self.text_fits_vertical(text, rect, format)
    }
    /// Determine the maximum font size for text with the given format
    /// that will still allow the text to fit within a rectangle
    fn fit_max_font_size<R, F>(&mut self, text: &str, rect: R, format: F) -> u32
    where
        R: Rectangle<Scalar = f64>,
        F: Into<TextFormat>,
    {
        let mut format = format.into();
        while !self.text_fits(text, rect, format) {
            format.font_size -= 1;
        }
        format.font_size
    }
    /// Determine the minumum height for a rectangle such that text
    /// with the given format will still fit within the rectangle
    ///
    /// The given delta value defines how much to increment the
    /// rectangle's height on each check. Lower deltas will yield
    /// more accurate results, but will take longer to computer.
    fn fit_min_height<R, F>(&mut self, text: &str, mut rect: R, format: F, delta: f64) -> f64
    where
        R: Rectangle<Scalar = f64>,
        F: Into<TextFormat>,
    {
        let format = format.into();
        let delta = delta.abs().max(1.0);
        while self.text_fits_vertical(text, rect, format) {
            rect = rect.with_size(R::Vector::new(rect.width(), rect.height() - delta))
        }
        while !self.text_fits_vertical(text, rect, format) {
            rect = rect.with_size(R::Vector::new(rect.width(), rect.height() + delta))
        }
        rect.height()
    }
    /// Determine the minumum width for a rectangle such that text
    /// with the given format will still fit within the rectangle
    ///
    /// The given delta value defines how much to increment the
    /// rectangle's width on each check. Lower deltas will yield
    /// more accurate results, but will take longer to computer.
    fn fit_min_width<R, F>(&mut self, text: &str, mut rect: R, format: F, delta: f64) -> f64
    where
        R: Rectangle<Scalar = f64>,
        F: Into<TextFormat>,
    {
        let format = format.into();
        let delta = delta.abs().max(1.0);
        while self.text_fits(text, rect, format) {
            rect = rect.with_size(R::Vector::new(rect.width() - delta, rect.height()))
        }
        while !self.text_fits(text, rect, format) {
            rect = rect.with_size(R::Vector::new(rect.width() + delta, rect.height()))
        }
        rect.width()
    }
}

/// A basic implememntor for `CharacterWidthCache`
#[derive(Clone)]
pub struct Glyphs<'f, S = f64>
where
    S: Scalar,
{
    widths: HashMap<(u32, char), S>,
    font: Font<'f>,
}

impl<'f, S> Glyphs<'f, S>
where
    S: Scalar,
{
    /// Loads a `Glyphs` from an array of font data.
    pub fn from_bytes(bytes: &'f [u8]) -> Result<Glyphs<'f, S>, Error> {
        Ok(Glyphs {
            widths: HashMap::new(),
            font: Font::from_bytes(bytes)?,
        })
    }
    /// Loads a `Glyphs` from a `Font`.
    pub fn from_font(font: Font<'f>) -> Glyphs<'f, S> {
        Glyphs {
            widths: HashMap::new(),
            font,
        }
    }
}

impl<'f> CharacterWidthCache for Glyphs<'f> {
    fn char_width(&mut self, character: char, font_size: u32) -> f64 {
        let font = &self.font;
        *self
            .widths
            .entry((font_size, character))
            .or_insert_with(|| {
                let scale = Scale::uniform(font_size as f32);
                let glyph = font.glyph(character).scaled(scale);
                let glyph = if glyph.id() == GlyphId(0) && glyph.shape().is_none() {
                    font.glyph('\u{FFFD}').scaled(scale)
                } else {
                    glyph
                };
                glyph.h_metrics().advance_width.into()
            })
    }
}

#[cfg(feature = "graphics")]
impl<C> CharacterWidthCache for C
where
    C: CharacterCache,
{
    fn char_width(&mut self, character: char, font_size: u32) -> f64 {
        if let Ok(character) = <Self as CharacterCache>::character(self, font_size, character) {
            character.texture.get_width() as f64
        } else {
            panic!("CharacterWidthCache::character returned Err")
        }
    }
}

/// Draw justified text to something using the `piston2d-graphics` crate
///
/// Text will be drawn in the given rectangle and use the given format
#[cfg(feature = "graphics")]
pub fn fitted_text<S, R, F, C, G>(
    text: S,
    rect: R,
    format: F,
    glyphs: &mut C,
    transform: Matrix2d,
    graphics: &mut G,
) -> Result<(), C::Error>
where
    S: AsRef<str>,
    R: Rectangle<Scalar = f64>,
    F: Into<TextFormat>,
    C: CharacterCache,
    C::Texture: ImageSize,
    G: Graphics<Texture = C::Texture>,
{
    let format = format.into();
    for (pos, line) in glyphs.justify_text(text.as_ref(), rect, format) {
        draw_text(
            format.color,
            format.font_size,
            &line,
            glyphs,
            transform.trans(pos.x(), pos.y()),
            graphics,
        )?;
    }
    Ok(())
}

/// A struct for writing text into multiple rectangles
#[cfg(feature = "graphics")]
pub struct Scribe<'a, C, G> {
    /// The text format
    pub format: TextFormat,
    /// The character cache
    pub glyphs: &'a mut C,
    /// The transform
    pub transform: Matrix2d,
    /// The graphics abstraction being drawn to
    pub graphics: &'a mut G,
}

#[cfg(feature = "graphics")]
impl<'a, C, G> Scribe<'a, C, G> {
    /// Create a new `Scribe`
    pub fn new(
        format: TextFormat,
        glyphs: &'a mut C,
        transform: Matrix2d,
        graphics: &'a mut G,
    ) -> Self {
        Scribe {
            format,
            glyphs,
            transform,
            graphics,
        }
    }
}

#[cfg(feature = "graphics")]
impl<'a, C, G> Scribe<'a, C, G>
where
    C: CharacterCache,
    G: Graphics<Texture = C::Texture>,
{
    /// Write some text into a rectangle with the scribe
    pub fn write<S, R>(&mut self, text: S, rectangle: R) -> Result<(), C::Error>
    where
        S: AsRef<str>,
        R: Rectangle<Scalar = f64>,
    {
        fitted_text(
            text.as_ref(),
            rectangle,
            self.format,
            self.glyphs,
            self.transform,
            self.graphics,
        )
    }
}

/// Defines serveral color constants
pub mod color {
    /// A 4-channel color
    pub type Color = [f32; 4];

    /// Red
    pub const RED: Color = [1.0, 0.0, 0.0, 1.0];
    /// Orange
    pub const ORANGE: Color = [1.0, 0.5, 0.0, 1.0];
    /// Yellow
    pub const YELLOW: Color = [1.0, 1.0, 0.0, 1.0];
    /// Green
    pub const GREEN: Color = [0.0, 1.0, 0.0, 1.0];
    /// Cyan
    pub const CYAN: Color = [0.0, 1.0, 1.0, 1.0];
    /// Blue
    pub const BLUE: Color = [0.0, 0.0, 1.0, 1.0];
    /// Purple
    pub const PURPLE: Color = [0.5, 0.0, 0.5, 1.0];
    /// Magenta
    pub const MAGENTA: Color = [1.0, 0.0, 1.0, 1.0];
    /// Black
    pub const BLACK: Color = [0.0, 0.0, 0.0, 1.0];
    /// Gray (same as `GREY`)
    pub const GRAY: Color = [0.5, 0.5, 0.5, 1.0];
    /// Grey (same as `GRAY`)
    pub const GREY: Color = GRAY;
    /// White
    pub const WHITE: Color = [1.0; 4];
    /// Transparent
    pub const TRANSPARENT: Color = [0.0; 4];
}

pub use self::color::Color;
