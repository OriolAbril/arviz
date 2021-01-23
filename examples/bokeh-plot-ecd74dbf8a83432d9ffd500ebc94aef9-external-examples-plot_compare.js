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
    
      
      
    
      var element = document.getElementById("a81f0bd7-773c-4582-82a8-804e9c587443");
        if (element == null) {
          console.warn("Bokeh: autoload.js configured with elementid 'a81f0bd7-773c-4582-82a8-804e9c587443' but no matching script tag was found.")
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
    
        const hashes = {"https://cdn.bokeh.org/bokeh/release/bokeh-2.2.3.min.js": "T2yuo9Oe71Cz/I4X9Ac5+gpEa5a8PpJCDlqKYO0CfAuEszu1JrXLl8YugMqYe3sM", "https://cdn.bokeh.org/bokeh/release/bokeh-widgets-2.2.3.min.js": "98GDGJ0kOMCUMUePhksaQ/GYgB3+NH9h996V88sh3aOiUNX3N+fLXAtry6xctSZ6", "https://cdn.bokeh.org/bokeh/release/bokeh-tables-2.2.3.min.js": "89bArO+nlbP3sgakeHjCo1JYxYR5wufVgA3IbUvDY+K7w4zyxJqssu7wVnfeKCq8"};
    
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
    
      
      var js_urls = ["https://cdn.bokeh.org/bokeh/release/bokeh-2.2.3.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-widgets-2.2.3.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-tables-2.2.3.min.js"];
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
                    
                  var docs_json = '{"a580782f-c2fb-482a-ae30-ceed91a1073a":{"roots":{"references":[{"attributes":{},"id":"14831","type":"LinearScale"},{"attributes":{},"id":"14900","type":"Selection"},{"attributes":{"data":{"x":{"__ndarray__":"KAWarnTPPsA=","dtype":"float64","order":"little","shape":[1]},"y":[-0.75]},"selected":{"id":"14898"},"selection_policy":{"id":"14897"}},"id":"14864","type":"ColumnDataSource"},{"attributes":{},"id":"14899","type":"UnionRenderers"},{"attributes":{"fill_color":{"value":"grey"},"line_color":{"value":"grey"},"line_width":{"value":2},"size":{"units":"screen","value":6},"x":{"field":"x"},"y":{"field":"y"}},"id":"14865","type":"Triangle"},{"attributes":{"data":{"x":{"__ndarray__":"m/f9Q2zYPcDPGP3dN9s9wA==","dtype":"float64","order":"little","shape":[2]},"y":[0.0,-1.0]},"selected":{"id":"14906"},"selection_policy":{"id":"14905"}},"id":"14884","type":"ColumnDataSource"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"14843"},{"id":"14844"},{"id":"14845"},{"id":"14846"},{"id":"14847"},{"id":"14848"},{"id":"14849"},{"id":"14850"}]},"id":"14853","type":"Toolbar"},{"attributes":{"source":{"id":"14884"}},"id":"14888","type":"CDSView"},{"attributes":{"line_alpha":{"value":0.1},"line_color":{"value":"grey"},"xs":{"field":"xs"},"ys":{"field":"ys"}},"id":"14871","type":"MultiLine"},{"attributes":{"toolbar":{"id":"14909"},"toolbar_location":"above"},"id":"14910","type":"ToolbarBox"},{"attributes":{},"id":"14843","type":"ResetTool"},{"attributes":{"formatter":{"id":"14894"},"major_label_overrides":{"-0.75":"","-1":"Centered 8 schools","0":"Non-centered 8 schools"},"ticker":{"id":"14862"}},"id":"14839","type":"LinearAxis"},{"attributes":{},"id":"14905","type":"UnionRenderers"},{"attributes":{},"id":"14849","type":"SaveTool"},{"attributes":{"xs":{"field":"xs"},"ys":{"field":"ys"}},"id":"14880","type":"MultiLine"},{"attributes":{"line_color":{"value":"grey"},"xs":{"field":"xs"},"ys":{"field":"ys"}},"id":"14870","type":"MultiLine"},{"attributes":{},"id":"14894","type":"BasicTickFormatter"},{"attributes":{},"id":"14846","type":"WheelZoomTool"},{"attributes":{"source":{"id":"14874"}},"id":"14878","type":"CDSView"},{"attributes":{"callback":null},"id":"14850","type":"HoverTool"},{"attributes":{},"id":"14906","type":"Selection"},{"attributes":{"source":{"id":"14879"}},"id":"14883","type":"CDSView"},{"attributes":{},"id":"14836","type":"BasicTicker"},{"attributes":{"axis":{"id":"14839"},"dimension":1,"ticker":null},"id":"14842","type":"Grid"},{"attributes":{"overlay":{"id":"14851"}},"id":"14845","type":"BoxZoomTool"},{"attributes":{"data_source":{"id":"14884"},"glyph":{"id":"14885"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"14886"},"selection_glyph":null,"view":{"id":"14888"}},"id":"14887","type":"GlyphRenderer"},{"attributes":{"overlay":{"id":"14852"}},"id":"14847","type":"LassoSelectTool"},{"attributes":{"data_source":{"id":"14869"},"glyph":{"id":"14870"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"14871"},"selection_glyph":null,"view":{"id":"14873"}},"id":"14872","type":"GlyphRenderer"},{"attributes":{"data":{"x":{"__ndarray__":"eFcgQvKvPsAoBZqudM8+wA==","dtype":"float64","order":"little","shape":[2]},"y":[0.0,-1.0]},"selected":{"id":"14902"},"selection_policy":{"id":"14901"}},"id":"14874","type":"ColumnDataSource"},{"attributes":{},"id":"14902","type":"Selection"},{"attributes":{"toolbars":[{"id":"14853"}],"tools":[{"id":"14843"},{"id":"14844"},{"id":"14845"},{"id":"14846"},{"id":"14847"},{"id":"14848"},{"id":"14849"},{"id":"14850"}]},"id":"14909","type":"ProxyToolbar"},{"attributes":{"data":{"xs":[[-30.896420573800537,-30.724327779399562]],"ys":[[-0.75,-0.75]]},"selected":{"id":"14900"},"selection_policy":{"id":"14899"}},"id":"14869","type":"ColumnDataSource"},{"attributes":{},"id":"14901","type":"UnionRenderers"},{"attributes":{"data_source":{"id":"14864"},"glyph":{"id":"14865"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"14866"},"selection_glyph":null,"view":{"id":"14868"}},"id":"14867","type":"GlyphRenderer"},{"attributes":{"source":{"id":"14864"}},"id":"14868","type":"CDSView"},{"attributes":{"line_alpha":{"value":0.1},"xs":{"field":"xs"},"ys":{"field":"ys"}},"id":"14881","type":"MultiLine"},{"attributes":{"text":""},"id":"14891","type":"Title"},{"attributes":{},"id":"14844","type":"PanTool"},{"attributes":{"data_source":{"id":"14874"},"glyph":{"id":"14875"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"14876"},"selection_glyph":null,"view":{"id":"14878"}},"id":"14877","type":"GlyphRenderer"},{"attributes":{"children":[{"id":"14910"},{"id":"14908"}]},"id":"14911","type":"Column"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"14851","type":"BoxAnnotation"},{"attributes":{"children":[[{"id":"14826"},0,0]]},"id":"14908","type":"GridBox"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"black"},"line_alpha":{"value":0.1},"line_width":{"value":2},"size":{"units":"screen","value":6},"x":{"field":"x"},"y":{"field":"y"}},"id":"14886","type":"Circle"},{"attributes":{},"id":"14903","type":"UnionRenderers"},{"attributes":{"fill_alpha":0.5,"fill_color":"lightgrey","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"xs_units":"screen","ys_units":"screen"},"id":"14852","type":"PolyAnnotation"},{"attributes":{"axis_label":"Log","formatter":{"id":"14893"},"ticker":{"id":"14836"}},"id":"14835","type":"LinearAxis"},{"attributes":{"fill_color":{"value":"black"},"line_width":{"value":2},"size":{"units":"screen","value":6},"x":{"field":"x"},"y":{"field":"y"}},"id":"14885","type":"Circle"},{"attributes":{},"id":"14898","type":"Selection"},{"attributes":{"data_source":{"id":"14879"},"glyph":{"id":"14880"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"14881"},"selection_glyph":null,"view":{"id":"14883"}},"id":"14882","type":"GlyphRenderer"},{"attributes":{"end":0.5,"start":-1.5},"id":"14829","type":"DataRange1d"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"grey"},"line_alpha":{"value":0.1},"line_color":{"value":"grey"},"line_width":{"value":2},"size":{"units":"screen","value":6},"x":{"field":"x"},"y":{"field":"y"}},"id":"14866","type":"Triangle"},{"attributes":{"fill_color":{"value":null},"line_width":{"value":2},"size":{"units":"screen","value":6},"x":{"field":"x"},"y":{"field":"y"}},"id":"14875","type":"Circle"},{"attributes":{},"id":"14848","type":"UndoTool"},{"attributes":{"below":[{"id":"14835"}],"center":[{"id":"14838"},{"id":"14842"}],"left":[{"id":"14839"}],"output_backend":"webgl","plot_height":500,"plot_width":500,"renderers":[{"id":"14867"},{"id":"14872"},{"id":"14877"},{"id":"14882"},{"id":"14887"},{"id":"14889"}],"title":{"id":"14891"},"toolbar":{"id":"14853"},"toolbar_location":null,"x_range":{"id":"14827"},"x_scale":{"id":"14831"},"y_range":{"id":"14829"},"y_scale":{"id":"14833"}},"id":"14826","subtype":"Figure","type":"Plot"},{"attributes":{},"id":"14897","type":"UnionRenderers"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":null},"line_alpha":{"value":0.1},"line_width":{"value":2},"size":{"units":"screen","value":6},"x":{"field":"x"},"y":{"field":"y"}},"id":"14876","type":"Circle"},{"attributes":{},"id":"14827","type":"DataRange1d"},{"attributes":{"source":{"id":"14869"}},"id":"14873","type":"CDSView"},{"attributes":{},"id":"14833","type":"LinearScale"},{"attributes":{},"id":"14904","type":"Selection"},{"attributes":{"axis":{"id":"14835"},"ticker":null},"id":"14838","type":"Grid"},{"attributes":{"dimension":"height","line_color":"grey","line_dash":[6],"line_width":1.7677669529663689,"location":-30.687290318389813},"id":"14889","type":"Span"},{"attributes":{},"id":"14893","type":"BasicTickFormatter"},{"attributes":{"ticks":[0.0,-0.75,-1.0]},"id":"14862","type":"FixedTicker"},{"attributes":{"data":{"xs":[[-32.052286212415325,-29.322294424364305],[-32.23721121836336,-29.38353713483674]],"ys":[[0.0,0.0],[-1.0,-1.0]]},"selected":{"id":"14904"},"selection_policy":{"id":"14903"}},"id":"14879","type":"ColumnDataSource"}],"root_ids":["14911"]},"title":"Bokeh Application","version":"2.2.3"}}';
                  var render_items = [{"docid":"a580782f-c2fb-482a-ae30-ceed91a1073a","root_ids":["14911"],"roots":{"14911":"a81f0bd7-773c-4582-82a8-804e9c587443"}}];
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