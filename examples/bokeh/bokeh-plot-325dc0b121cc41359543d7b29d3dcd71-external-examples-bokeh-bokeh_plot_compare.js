(function() {
  var fn = function() {
    
    (function(root) {
      function now() {
        return new Date();
      }
    
      var force = false;
    
      if (typeof root._bokeh_onload_callbacks === "undefined" || force === true) {
        root._bokeh_onload_callbacks = [];
        root._bokeh_is_loading = undefined;
      }
    
      
      
    
      var element = document.getElementById("f081fac1-e340-4bd1-9f9f-e251d92700d3");
        if (element == null) {
          console.warn("Bokeh: autoload.js configured with elementid 'f081fac1-e340-4bd1-9f9f-e251d92700d3' but no matching script tag was found.")
        }
      
    
      function run_callbacks() {
        try {
          root._bokeh_onload_callbacks.forEach(function(callback) {
            if (callback != null)
              callback();
          });
        } finally {
          delete root._bokeh_onload_callbacks
        }
        console.debug("Bokeh: all callbacks have finished");
      }
    
      function load_libs(css_urls, js_urls, callback) {
        if (css_urls == null) css_urls = [];
        if (js_urls == null) js_urls = [];
    
        root._bokeh_onload_callbacks.push(callback);
        if (root._bokeh_is_loading > 0) {
          console.debug("Bokeh: BokehJS is being loaded, scheduling callback at", now());
          return null;
        }
        if (js_urls == null || js_urls.length === 0) {
          run_callbacks();
          return null;
        }
        console.debug("Bokeh: BokehJS not loaded, scheduling load and callback at", now());
        root._bokeh_is_loading = css_urls.length + js_urls.length;
    
        function on_load() {
          root._bokeh_is_loading--;
          if (root._bokeh_is_loading === 0) {
            console.debug("Bokeh: all BokehJS libraries/stylesheets loaded");
            run_callbacks()
          }
        }
    
        function on_error() {
          console.error("failed to load " + url);
        }
    
        for (var i = 0; i < css_urls.length; i++) {
          var url = css_urls[i];
          const element = document.createElement("link");
          element.onload = on_load;
          element.onerror = on_error;
          element.rel = "stylesheet";
          element.type = "text/css";
          element.href = url;
          console.debug("Bokeh: injecting link tag for BokehJS stylesheet: ", url);
          document.body.appendChild(element);
        }
    
        const hashes = {"https://cdn.bokeh.org/bokeh/release/bokeh-2.2.1.min.js": "qkRvDQVAIfzsJo40iRBbxt6sttt0hv4lh74DG7OK4MCHv4C5oohXYoHUM5W11uqS", "https://cdn.bokeh.org/bokeh/release/bokeh-widgets-2.2.1.min.js": "Sb7Mr06a9TNlet/GEBeKaf5xH3eb6AlCzwjtU82wNPyDrnfoiVl26qnvlKjmcAd+", "https://cdn.bokeh.org/bokeh/release/bokeh-tables-2.2.1.min.js": "HaJ15vgfmcfRtB4c4YBOI4f1MUujukqInOWVqZJZZGK7Q+ivud0OKGSTn/Vm2iso"};
    
        for (var i = 0; i < js_urls.length; i++) {
          var url = js_urls[i];
          var element = document.createElement('script');
          element.onload = on_load;
          element.onerror = on_error;
          element.async = false;
          element.src = url;
          if (url in hashes) {
            element.crossOrigin = "anonymous";
            element.integrity = "sha384-" + hashes[url];
          }
          console.debug("Bokeh: injecting script tag for BokehJS library: ", url);
          document.head.appendChild(element);
        }
      };
    
      function inject_raw_css(css) {
        const element = document.createElement("style");
        element.appendChild(document.createTextNode(css));
        document.body.appendChild(element);
      }
    
      
      var js_urls = ["https://cdn.bokeh.org/bokeh/release/bokeh-2.2.1.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-widgets-2.2.1.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-tables-2.2.1.min.js"];
      var css_urls = [];
      
    
      var inline_js = [
        function(Bokeh) {
          Bokeh.set_log_level("info");
        },
        
        function(Bokeh) {
          (function() {
            var fn = function() {
              Bokeh.safely(function() {
                (function(root) {
                  function embed_document(root) {
                    
                  var docs_json = '{"68676a63-bf46-4dec-bab5-5c4af47f38e8":{"roots":{"references":[{"attributes":{"end":0.5,"start":-1.5},"id":"1773","type":"DataRange1d"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"1795","type":"BoxAnnotation"},{"attributes":{},"id":"1787","type":"ResetTool"},{"attributes":{"data_source":{"id":"1818"},"glyph":{"id":"1819"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"1820"},"selection_glyph":null,"view":{"id":"1822"}},"id":"1821","type":"GlyphRenderer"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":null},"line_alpha":{"value":0.1},"line_width":{"value":2},"size":{"units":"screen","value":6},"x":{"field":"x"},"y":{"field":"y"}},"id":"1820","type":"Circle"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"1787"},{"id":"1788"},{"id":"1789"},{"id":"1790"},{"id":"1791"},{"id":"1792"},{"id":"1793"},{"id":"1794"}]},"id":"1797","type":"Toolbar"},{"attributes":{"data":{"xs":[[-32.0245964160785,-29.349984220701124],[-32.09039664183325,-29.530351711366848]],"ys":[[0.0,0.0],[-1.0,-1.0]]},"selected":{"id":"1847"},"selection_policy":{"id":"1848"}},"id":"1823","type":"ColumnDataSource"},{"attributes":{},"id":"1847","type":"Selection"},{"attributes":{"axis_label":"Log","formatter":{"id":"1838"},"ticker":{"id":"1780"}},"id":"1779","type":"LinearAxis"},{"attributes":{},"id":"1836","type":"BasicTickFormatter"},{"attributes":{"line_alpha":{"value":0.1},"xs":{"field":"xs"},"ys":{"field":"ys"}},"id":"1825","type":"MultiLine"},{"attributes":{},"id":"1792","type":"UndoTool"},{"attributes":{"xs":{"field":"xs"},"ys":{"field":"ys"}},"id":"1824","type":"MultiLine"},{"attributes":{"overlay":{"id":"1795"}},"id":"1789","type":"BoxZoomTool"},{"attributes":{"fill_color":{"value":null},"line_width":{"value":2},"size":{"units":"screen","value":6},"x":{"field":"x"},"y":{"field":"y"}},"id":"1819","type":"Circle"},{"attributes":{},"id":"1844","type":"UnionRenderers"},{"attributes":{"source":{"id":"1813"}},"id":"1817","type":"CDSView"},{"attributes":{},"id":"1780","type":"BasicTicker"},{"attributes":{"overlay":{"id":"1796"}},"id":"1791","type":"LassoSelectTool"},{"attributes":{"fill_color":{"value":"black"},"line_width":{"value":2},"size":{"units":"screen","value":6},"x":{"field":"x"},"y":{"field":"y"}},"id":"1829","type":"Circle"},{"attributes":{"fill_alpha":0.5,"fill_color":"lightgrey","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"xs_units":"screen","ys_units":"screen"},"id":"1796","type":"PolyAnnotation"},{"attributes":{},"id":"1841","type":"Selection"},{"attributes":{},"id":"1771","type":"DataRange1d"},{"attributes":{"data_source":{"id":"1813"},"glyph":{"id":"1814"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"1815"},"selection_glyph":null,"view":{"id":"1817"}},"id":"1816","type":"GlyphRenderer"},{"attributes":{"fill_color":{"value":"grey"},"line_color":{"value":"grey"},"line_width":{"value":2},"size":{"units":"screen","value":6},"x":{"field":"x"},"y":{"field":"y"}},"id":"1809","type":"Triangle"},{"attributes":{"children":[[{"id":"1770"},0,0]]},"id":"1852","type":"GridBox"},{"attributes":{},"id":"1846","type":"UnionRenderers"},{"attributes":{},"id":"1848","type":"UnionRenderers"},{"attributes":{"data_source":{"id":"1808"},"glyph":{"id":"1809"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"1810"},"selection_glyph":null,"view":{"id":"1812"}},"id":"1811","type":"GlyphRenderer"},{"attributes":{"data_source":{"id":"1828"},"glyph":{"id":"1829"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"1830"},"selection_glyph":null,"view":{"id":"1832"}},"id":"1831","type":"GlyphRenderer"},{"attributes":{},"id":"1843","type":"Selection"},{"attributes":{"callback":null},"id":"1794","type":"HoverTool"},{"attributes":{},"id":"1788","type":"PanTool"},{"attributes":{"line_color":{"value":"grey"},"xs":{"field":"xs"},"ys":{"field":"ys"}},"id":"1814","type":"MultiLine"},{"attributes":{"source":{"id":"1808"}},"id":"1812","type":"CDSView"},{"attributes":{"dimension":"height","line_color":"grey","line_dash":[6],"line_width":1.875,"location":-30.687290318389813},"id":"1833","type":"Span"},{"attributes":{"children":[{"id":"1854"},{"id":"1852"}]},"id":"1855","type":"Column"},{"attributes":{"axis":{"id":"1779"},"ticker":null},"id":"1782","type":"Grid"},{"attributes":{"data_source":{"id":"1823"},"glyph":{"id":"1824"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"1825"},"selection_glyph":null,"view":{"id":"1827"}},"id":"1826","type":"GlyphRenderer"},{"attributes":{"formatter":{"id":"1836"},"major_label_overrides":{"-0.75":"","-1":"Centered 8 schools","0":"Non-centered 8 schools"},"ticker":{"id":"1806"}},"id":"1783","type":"LinearAxis"},{"attributes":{"text":""},"id":"1834","type":"Title"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"black"},"line_alpha":{"value":0.1},"line_width":{"value":2},"size":{"units":"screen","value":6},"x":{"field":"x"},"y":{"field":"y"}},"id":"1830","type":"Circle"},{"attributes":{},"id":"1790","type":"WheelZoomTool"},{"attributes":{"below":[{"id":"1779"}],"center":[{"id":"1782"},{"id":"1786"}],"left":[{"id":"1783"}],"output_backend":"webgl","plot_height":500,"plot_width":500,"renderers":[{"id":"1811"},{"id":"1816"},{"id":"1821"},{"id":"1826"},{"id":"1831"},{"id":"1833"}],"title":{"id":"1834"},"toolbar":{"id":"1797"},"toolbar_location":null,"x_range":{"id":"1771"},"x_scale":{"id":"1775"},"y_range":{"id":"1773"},"y_scale":{"id":"1777"}},"id":"1770","subtype":"Figure","type":"Plot"},{"attributes":{},"id":"1845","type":"Selection"},{"attributes":{"source":{"id":"1818"}},"id":"1822","type":"CDSView"},{"attributes":{"toolbars":[{"id":"1797"}],"tools":[{"id":"1787"},{"id":"1788"},{"id":"1789"},{"id":"1790"},{"id":"1791"},{"id":"1792"},{"id":"1793"},{"id":"1794"}]},"id":"1853","type":"ProxyToolbar"},{"attributes":{"data":{"x":[-30.81037417660005],"y":[-0.75]},"selected":{"id":"1841"},"selection_policy":{"id":"1842"}},"id":"1808","type":"ColumnDataSource"},{"attributes":{"data":{"x":[-30.687290318389813,-30.81037417660005],"y":[0.0,-1.0]},"selected":{"id":"1845"},"selection_policy":{"id":"1846"}},"id":"1818","type":"ColumnDataSource"},{"attributes":{"toolbar":{"id":"1853"},"toolbar_location":"above"},"id":"1854","type":"ToolbarBox"},{"attributes":{"line_alpha":{"value":0.1},"line_color":{"value":"grey"},"xs":{"field":"xs"},"ys":{"field":"ys"}},"id":"1815","type":"MultiLine"},{"attributes":{"source":{"id":"1823"}},"id":"1827","type":"CDSView"},{"attributes":{"data":{"xs":[[-30.896420573800537,-30.724327779399562]],"ys":[[-0.75,-0.75]]},"selected":{"id":"1843"},"selection_policy":{"id":"1844"}},"id":"1813","type":"ColumnDataSource"},{"attributes":{},"id":"1775","type":"LinearScale"},{"attributes":{"data":{"x":[-32.37106695144684,-32.71848009989285],"y":[0.0,-1.0]},"selected":{"id":"1849"},"selection_policy":{"id":"1850"}},"id":"1828","type":"ColumnDataSource"},{"attributes":{},"id":"1842","type":"UnionRenderers"},{"attributes":{},"id":"1850","type":"UnionRenderers"},{"attributes":{"ticks":[0.0,-0.75,-1.0]},"id":"1806","type":"FixedTicker"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"grey"},"line_alpha":{"value":0.1},"line_color":{"value":"grey"},"line_width":{"value":2},"size":{"units":"screen","value":6},"x":{"field":"x"},"y":{"field":"y"}},"id":"1810","type":"Triangle"},{"attributes":{"source":{"id":"1828"}},"id":"1832","type":"CDSView"},{"attributes":{"axis":{"id":"1783"},"dimension":1,"ticker":null},"id":"1786","type":"Grid"},{"attributes":{},"id":"1777","type":"LinearScale"},{"attributes":{},"id":"1849","type":"Selection"},{"attributes":{},"id":"1793","type":"SaveTool"},{"attributes":{},"id":"1838","type":"BasicTickFormatter"}],"root_ids":["1855"]},"title":"Bokeh Application","version":"2.2.1"}}';
                  var render_items = [{"docid":"68676a63-bf46-4dec-bab5-5c4af47f38e8","root_ids":["1855"],"roots":{"1855":"f081fac1-e340-4bd1-9f9f-e251d92700d3"}}];
                  root.Bokeh.embed.embed_items(docs_json, render_items);
                
                  }
                  if (root.Bokeh !== undefined) {
                    embed_document(root);
                  } else {
                    var attempts = 0;
                    var timer = setInterval(function(root) {
                      if (root.Bokeh !== undefined) {
                        clearInterval(timer);
                        embed_document(root);
                      } else {
                        attempts++;
                        if (attempts > 100) {
                          clearInterval(timer);
                          console.log("Bokeh: ERROR: Unable to run BokehJS code because BokehJS library is missing");
                        }
                      }
                    }, 10, root)
                  }
                })(window);
              });
            };
            if (document.readyState != "loading") fn();
            else document.addEventListener("DOMContentLoaded", fn);
          })();
        },
        function(Bokeh) {
        
        
        }
      ];
    
      function run_inline_js() {
        
        for (var i = 0; i < inline_js.length; i++) {
          inline_js[i].call(root, root.Bokeh);
        }
        
      }
    
      if (root._bokeh_is_loading === 0) {
        console.debug("Bokeh: BokehJS loaded, going straight to plotting");
        run_inline_js();
      } else {
        load_libs(css_urls, js_urls, function() {
          console.debug("Bokeh: BokehJS plotting callback run at", now());
          run_inline_js();
        });
      }
    }(window));
  };
  if (document.readyState != "loading") fn();
  else document.addEventListener("DOMContentLoaded", fn);
})();