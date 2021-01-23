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
    
      
      
    
      var element = document.getElementById("2342df86-6a07-4cf0-8bcc-bfd6a85fb8f8");
        if (element == null) {
          console.warn("Bokeh: autoload.js configured with elementid '2342df86-6a07-4cf0-8bcc-bfd6a85fb8f8' but no matching script tag was found.")
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
                    
                  var docs_json = '{"84adf5fc-acd1-4cfd-a205-d3b47ccc16f7":{"roots":{"references":[{"attributes":{"data_source":{"id":"1736"},"glyph":{"id":"1737"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"1738"},"selection_glyph":null,"view":{"id":"1740"}},"id":"1739","type":"GlyphRenderer"},{"attributes":{},"id":"1744","type":"BasicTickFormatter"},{"attributes":{"source":{"id":"1721"}},"id":"1725","type":"CDSView"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":null},"line_alpha":{"value":0.1},"line_width":{"value":2},"size":{"units":"screen","value":6},"x":{"field":"x"},"y":{"field":"y"}},"id":"1728","type":"Circle"},{"attributes":{"data_source":{"id":"1721"},"glyph":{"id":"1722"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"1723"},"selection_glyph":null,"view":{"id":"1725"}},"id":"1724","type":"GlyphRenderer"},{"attributes":{"children":[{"id":"1762"},{"id":"1760"}]},"id":"1763","type":"Column"},{"attributes":{"overlay":{"id":"1704"}},"id":"1699","type":"LassoSelectTool"},{"attributes":{"source":{"id":"1726"}},"id":"1730","type":"CDSView"},{"attributes":{"toolbars":[{"id":"1705"}],"tools":[{"id":"1695"},{"id":"1696"},{"id":"1697"},{"id":"1698"},{"id":"1699"},{"id":"1700"},{"id":"1701"},{"id":"1702"}]},"id":"1761","type":"ProxyToolbar"},{"attributes":{"xs":{"field":"xs"},"ys":{"field":"ys"}},"id":"1732","type":"MultiLine"},{"attributes":{"dimension":"height","line_color":"grey","line_dash":[6],"line_width":1.875,"location":-30.687290318389813},"id":"1741","type":"Span"},{"attributes":{"data":{"xs":[[-30.896420573800537,-30.724327779399562]],"ys":[[-0.75,-0.75]]},"selected":{"id":"1751"},"selection_policy":{"id":"1752"}},"id":"1721","type":"ColumnDataSource"},{"attributes":{},"id":"1688","type":"BasicTicker"},{"attributes":{"data":{"x":{"__ndarray__":"eFcgQvKvPsAoBZqudM8+wA==","dtype":"float64","order":"little","shape":[2]},"y":[0.0,-1.0]},"selected":{"id":"1753"},"selection_policy":{"id":"1754"}},"id":"1726","type":"ColumnDataSource"},{"attributes":{"source":{"id":"1736"}},"id":"1740","type":"CDSView"},{"attributes":{"line_color":{"value":"grey"},"xs":{"field":"xs"},"ys":{"field":"ys"}},"id":"1722","type":"MultiLine"},{"attributes":{"line_alpha":{"value":0.1},"line_color":{"value":"grey"},"xs":{"field":"xs"},"ys":{"field":"ys"}},"id":"1723","type":"MultiLine"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"black"},"line_alpha":{"value":0.1},"line_width":{"value":2},"size":{"units":"screen","value":6},"x":{"field":"x"},"y":{"field":"y"}},"id":"1738","type":"Circle"},{"attributes":{},"id":"1746","type":"BasicTickFormatter"},{"attributes":{"formatter":{"id":"1744"},"major_label_overrides":{"-0.75":"","-1":"Centered 8 schools","0":"Non-centered 8 schools"},"ticker":{"id":"1714"}},"id":"1691","type":"LinearAxis"},{"attributes":{},"id":"1698","type":"WheelZoomTool"},{"attributes":{"overlay":{"id":"1703"}},"id":"1697","type":"BoxZoomTool"},{"attributes":{"data":{"x":{"__ndarray__":"KAWarnTPPsA=","dtype":"float64","order":"little","shape":[1]},"y":[-0.75]},"selected":{"id":"1749"},"selection_policy":{"id":"1750"}},"id":"1716","type":"ColumnDataSource"},{"attributes":{"fill_color":{"value":null},"line_width":{"value":2},"size":{"units":"screen","value":6},"x":{"field":"x"},"y":{"field":"y"}},"id":"1727","type":"Circle"},{"attributes":{"toolbar":{"id":"1761"},"toolbar_location":"above"},"id":"1762","type":"ToolbarBox"},{"attributes":{"data_source":{"id":"1726"},"glyph":{"id":"1727"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"1728"},"selection_glyph":null,"view":{"id":"1730"}},"id":"1729","type":"GlyphRenderer"},{"attributes":{"line_alpha":{"value":0.1},"xs":{"field":"xs"},"ys":{"field":"ys"}},"id":"1733","type":"MultiLine"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"1703","type":"BoxAnnotation"},{"attributes":{"data_source":{"id":"1731"},"glyph":{"id":"1732"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"1733"},"selection_glyph":null,"view":{"id":"1735"}},"id":"1734","type":"GlyphRenderer"},{"attributes":{},"id":"1700","type":"UndoTool"},{"attributes":{"data":{"xs":[[-32.052286212415325,-29.322294424364305],[-32.23721121836336,-29.38353713483674]],"ys":[[0.0,0.0],[-1.0,-1.0]]},"selected":{"id":"1755"},"selection_policy":{"id":"1756"}},"id":"1731","type":"ColumnDataSource"},{"attributes":{"source":{"id":"1731"}},"id":"1735","type":"CDSView"},{"attributes":{},"id":"1701","type":"SaveTool"},{"attributes":{"data_source":{"id":"1716"},"glyph":{"id":"1717"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"1718"},"selection_glyph":null,"view":{"id":"1720"}},"id":"1719","type":"GlyphRenderer"},{"attributes":{"axis_label":"Log","formatter":{"id":"1746"},"ticker":{"id":"1688"}},"id":"1687","type":"LinearAxis"},{"attributes":{},"id":"1755","type":"Selection"},{"attributes":{"data":{"x":{"__ndarray__":"m/f9Q2zYPcDPGP3dN9s9wA==","dtype":"float64","order":"little","shape":[2]},"y":[0.0,-1.0]},"selected":{"id":"1757"},"selection_policy":{"id":"1758"}},"id":"1736","type":"ColumnDataSource"},{"attributes":{},"id":"1695","type":"ResetTool"},{"attributes":{"fill_alpha":0.5,"fill_color":"lightgrey","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"xs_units":"screen","ys_units":"screen"},"id":"1704","type":"PolyAnnotation"},{"attributes":{},"id":"1757","type":"Selection"},{"attributes":{},"id":"1750","type":"UnionRenderers"},{"attributes":{},"id":"1758","type":"UnionRenderers"},{"attributes":{"fill_color":{"value":"grey"},"line_color":{"value":"grey"},"line_width":{"value":2},"size":{"units":"screen","value":6},"x":{"field":"x"},"y":{"field":"y"}},"id":"1717","type":"Triangle"},{"attributes":{"source":{"id":"1716"}},"id":"1720","type":"CDSView"},{"attributes":{},"id":"1683","type":"LinearScale"},{"attributes":{},"id":"1696","type":"PanTool"},{"attributes":{},"id":"1751","type":"Selection"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"1695"},{"id":"1696"},{"id":"1697"},{"id":"1698"},{"id":"1699"},{"id":"1700"},{"id":"1701"},{"id":"1702"}]},"id":"1705","type":"Toolbar"},{"attributes":{"children":[[{"id":"1678"},0,0]]},"id":"1760","type":"GridBox"},{"attributes":{"end":0.5,"start":-1.5},"id":"1681","type":"DataRange1d"},{"attributes":{},"id":"1752","type":"UnionRenderers"},{"attributes":{"axis":{"id":"1687"},"ticker":null},"id":"1690","type":"Grid"},{"attributes":{"callback":null},"id":"1702","type":"HoverTool"},{"attributes":{},"id":"1685","type":"LinearScale"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"grey"},"line_alpha":{"value":0.1},"line_color":{"value":"grey"},"line_width":{"value":2},"size":{"units":"screen","value":6},"x":{"field":"x"},"y":{"field":"y"}},"id":"1718","type":"Triangle"},{"attributes":{},"id":"1749","type":"Selection"},{"attributes":{"axis":{"id":"1691"},"dimension":1,"ticker":null},"id":"1694","type":"Grid"},{"attributes":{"fill_color":{"value":"black"},"line_width":{"value":2},"size":{"units":"screen","value":6},"x":{"field":"x"},"y":{"field":"y"}},"id":"1737","type":"Circle"},{"attributes":{"ticks":[0.0,-0.75,-1.0]},"id":"1714","type":"FixedTicker"},{"attributes":{},"id":"1756","type":"UnionRenderers"},{"attributes":{"text":""},"id":"1742","type":"Title"},{"attributes":{},"id":"1753","type":"Selection"},{"attributes":{},"id":"1679","type":"DataRange1d"},{"attributes":{},"id":"1754","type":"UnionRenderers"},{"attributes":{"below":[{"id":"1687"}],"center":[{"id":"1690"},{"id":"1694"}],"left":[{"id":"1691"}],"output_backend":"webgl","plot_height":500,"plot_width":500,"renderers":[{"id":"1719"},{"id":"1724"},{"id":"1729"},{"id":"1734"},{"id":"1739"},{"id":"1741"}],"title":{"id":"1742"},"toolbar":{"id":"1705"},"toolbar_location":null,"x_range":{"id":"1679"},"x_scale":{"id":"1683"},"y_range":{"id":"1681"},"y_scale":{"id":"1685"}},"id":"1678","subtype":"Figure","type":"Plot"}],"root_ids":["1763"]},"title":"Bokeh Application","version":"2.2.3"}}';
                  var render_items = [{"docid":"84adf5fc-acd1-4cfd-a205-d3b47ccc16f7","root_ids":["1763"],"roots":{"1763":"2342df86-6a07-4cf0-8bcc-bfd6a85fb8f8"}}];
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